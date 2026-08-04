// Copyright 2025 edge_vlm_ros contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Integration-style tests for VlmReasonerNode using FakeInferenceBackend.
/// No NVIDIA hardware required.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "edge_vlm_ros/msg/tracked_observation.hpp"
#include "edge_vlm_ros/vlm_reasoner_node.hpp"
#include "edge_vlm_ros/msg/vlm_result.hpp"
#include "fake_inference_backend.hpp"

using namespace std::chrono_literals;
using edge_vlm_ros::VlmReasonerNode;
using edge_vlm_ros::FakeInferenceBackend;
using edge_vlm_ros::FailingFakeInferenceBackend;
using edge_vlm_ros::HistoryEntry;
using edge_vlm_ros::InferenceRequest;
using edge_vlm_ros::InferenceResponse;
using edge_vlm_ros::SlowFakeInferenceBackend;
using ResultMsg = edge_vlm_ros::msg::VlmResult;

// ─── helpers ─────────────────────────────────────────────────────────────────

static sensor_msgs::msg::Image::SharedPtr make_image(rclcpp::Time stamp)
{
  auto img = std::make_shared<sensor_msgs::msg::Image>();
  img->header.stamp = stamp;
  img->encoding = "bgr8";
  img->width = 4;
  img->height = 4;
  img->step = 4 * 3;
  img->data.assign(4 * 4 * 3, 200u);
  return img;
}

static edge_vlm_ros::msg::TrackedObservation::SharedPtr make_tracked_observation(
  rclcpp::Time stamp, uint64_t source_sequence = 1)
{
  auto observation = std::make_shared<edge_vlm_ros::msg::TrackedObservation>();
  observation->header.stamp = stamp;
  observation->source_stamp = stamp;
  observation->source_topic = "/camera/image_raw";
  observation->detector_id = "detector";
  observation->tracker_id = "tracker";
  observation->source_sequence = source_sequence;
  observation->source_image = *make_image(stamp);

  edge_vlm_ros::msg::TrackedObject tracked;
  tracked.header.stamp = stamp;
  tracked.track_id = 7;
  tracked.class_label = "person";
  tracked.confidence = 0.9f;
  tracked.center_x = 10.0f;
  tracked.center_y = 12.0f;
  tracked.width = 4.0f;
  tracked.height = 5.0f;
  tracked.age = 2;
  tracked.coast_age = 0;
  observation->tracked_objects.push_back(tracked);
  return observation;
}

static rclcpp::NodeOptions make_options(bool publish = true)
{
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("llm_engine_dir", "/nonexistent");
  opts.append_parameter_override("multimodal_engine_dir", "/nonexistent");
  opts.append_parameter_override("edge_llm_plugin_path", "/nonexistent");
  opts.append_parameter_override("sample_period_seconds", 0.05);  // 50 ms – fast sampling
  opts.append_parameter_override("publish_results", publish);
  return opts;
}

/// Spin two nodes until a predicate is satisfied or the timeout expires.
static bool spin_until(
  rclcpp::Node::SharedPtr n1,
  rclcpp::Node::SharedPtr n2,
  std::function<bool()> pred,
  std::chrono::milliseconds timeout = 3s)
{
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!pred() && std::chrono::steady_clock::now() < deadline) {
    rclcpp::spin_some(n1);
    if (n2) {
      rclcpp::spin_some(n2);
    }
    std::this_thread::sleep_for(10ms);
  }
  return pred();
}

// ─── Test fixture ─────────────────────────────────────────────────────────────

class NodeTest : public ::testing::Test
{
protected:
  void SetUp() override {}
  void TearDown() override
  {
    node_.reset();
  }

  std::shared_ptr<VlmReasonerNode> node_;
};

// ─── Tests ────────────────────────────────────────────────────────────────────

/// After construction the fake backend is initialized.
TEST_F(NodeTest, BackendInitialized)
{
  auto * raw_backend = new FakeInferenceBackend();
  node_ = std::make_shared<VlmReasonerNode>(
    std::unique_ptr<FakeInferenceBackend>(raw_backend),
    make_options(false));
  EXPECT_TRUE(raw_backend->is_initialized());
}

/// A successful fake inference publishes a result with success == true.
TEST_F(NodeTest, SuccessfulInferencePublishesResult)
{
  std::atomic<bool> received{false};
  ResultMsg last_msg;

  auto helper = std::make_shared<rclcpp::Node>("_test_helper_success");

  // Subscribe to results
  auto sub = helper->create_subscription<ResultMsg>(
    "/vlm/result", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      last_msg = *msg;
      received = true;
    });

  node_ = std::make_shared<VlmReasonerNode>(
    std::make_unique<FakeInferenceBackend>(),
    make_options(true));

  // Give DDS time to connect
  std::this_thread::sleep_for(100ms);

  // Publish an image using matching QoS
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = helper->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);

  std::this_thread::sleep_for(50ms);
  pub->publish(*make_image(rclcpp::Time(100, 0, RCL_ROS_TIME)));

  bool ok = spin_until(node_, helper, [&] {return received.load();});

  ASSERT_TRUE(ok) << "No result message received within timeout";
  EXPECT_TRUE(last_msg.success);
  EXPECT_EQ(last_msg.response, "fake response");
  EXPECT_FALSE(last_msg.source_topic.empty());
}

/// A failing backend publishes success == false and preserves the error string.
TEST_F(NodeTest, FailingInferencePublishesError)
{
  std::atomic<bool> received{false};
  ResultMsg last_msg;

  auto helper = std::make_shared<rclcpp::Node>("_test_helper_fail");
  auto sub = helper->create_subscription<ResultMsg>(
    "/vlm/result", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      last_msg = *msg;
      received = true;
    });

  node_ = std::make_shared<VlmReasonerNode>(
    std::make_unique<FailingFakeInferenceBackend>(),
    make_options(true));

  std::this_thread::sleep_for(100ms);

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = helper->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);

  std::this_thread::sleep_for(50ms);
  pub->publish(*make_image(rclcpp::Time(200, 0, RCL_ROS_TIME)));

  bool ok = spin_until(node_, helper, [&] {return received.load();});

  ASSERT_TRUE(ok) << "No result message received within timeout";
  EXPECT_FALSE(last_msg.success);
  EXPECT_FALSE(last_msg.error.empty());

  // Node must still be alive after a single failure
  EXPECT_TRUE(rclcpp::ok());
}

/// The subscription callback must not block even when the backend is slow.
TEST_F(NodeTest, CallbackDoesNotBlockOnSlowBackend)
{
  constexpr auto kSlowDelay = 300ms;

  node_ = std::make_shared<VlmReasonerNode>(
    std::make_unique<SlowFakeInferenceBackend>(kSlowDelay),
    make_options(false));

  std::this_thread::sleep_for(50ms);

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);

  std::this_thread::sleep_for(50ms);

  auto t0 = std::chrono::steady_clock::now();
  pub->publish(*make_image(rclcpp::Time(300, 0, RCL_ROS_TIME)));
  rclcpp::spin_some(node_);
  auto callback_time = std::chrono::steady_clock::now() - t0;

  // Callback itself + spin_some overhead must be well under the slow delay
  EXPECT_LT(callback_time, kSlowDelay / 2)
    << "spin_some took "
    << std::chrono::duration_cast<std::chrono::milliseconds>(callback_time).count()
    << " ms — appears to be blocking on inference";
}

/// When drop_old_frames == true and the worker is busy, the older pending
/// frame is replaced with the newer one (drop counter increments).
TEST_F(NodeTest, FullQueueDropsOldFrameWhenConfigured)
{
  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("drop_old_frames", true);
  // Use a slow backend (500 ms) so the worker can't keep up
  node_ = std::make_shared<VlmReasonerNode>(
    std::make_unique<SlowFakeInferenceBackend>(500ms),
    opts);

  std::this_thread::sleep_for(50ms);

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);

  std::this_thread::sleep_for(50ms);

  // Publish three frames 200 ms apart (period = 50 ms, so all three are sampled)
  for (int i = 0; i < 3; ++i) {
    int64_t ns = static_cast<int64_t>(400) * 1'000'000 + i * 200'000'000;
    int64_t sec = ns / 1'000'000'000;
    uint32_t nsec = static_cast<uint32_t>(ns % 1'000'000'000);
    pub->publish(*make_image(rclcpp::Time(sec, nsec, RCL_ROS_TIME)));
    rclcpp::spin_some(node_);
    std::this_thread::sleep_for(30ms);
  }

  // At least one frame should have been dropped because the worker is busy
  EXPECT_GE(node_->dropped_count(), 1u)
    << "Expected at least one dropped frame when worker is slow";
}

/// The node shuts down cleanly: the worker thread is joined without deadlock.
TEST_F(NodeTest, CleanShutdown)
{
  node_ = std::make_shared<VlmReasonerNode>(
    std::make_unique<FakeInferenceBackend>(),
    make_options(false));

  node_.reset();  // destructor joins worker thread
  SUCCEED();
}

/// Invalid sample_period_seconds is rejected during construction.
TEST_F(NodeTest, InvalidSamplePeriodRejected)
{
  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("sample_period_seconds", -1.0);

  EXPECT_THROW(
    std::make_shared<VlmReasonerNode>(
      std::make_unique<FakeInferenceBackend>(), opts),
    std::exception);
}

TEST_F(NodeTest, ZeroSamplePeriodIsAllowed)
{
  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("sample_period_seconds", 0.0);

  EXPECT_NO_THROW(
    node_ = std::make_shared<VlmReasonerNode>(
      std::make_unique<FakeInferenceBackend>(), opts));
}

/// Invalid temperature is rejected during construction.
TEST_F(NodeTest, InvalidTemperatureRejected)
{
  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("temperature", -0.5);

  EXPECT_THROW(
    std::make_shared<VlmReasonerNode>(
      std::make_unique<FakeInferenceBackend>(), opts),
    std::exception);
}

/// Selected task profile and prompt version are recorded with each published result.
TEST_F(NodeTest, PublishesProfileAndPromptVersionMetadata)
{
  std::atomic<bool> received{false};
  ResultMsg last_msg;

  auto helper = std::make_shared<rclcpp::Node>("_test_helper_profile_metadata");
  auto sub = helper->create_subscription<ResultMsg>(
    "/vlm/result", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      last_msg = *msg;
      received = true;
    });

  rclcpp::NodeOptions opts = make_options(true);
  opts.append_parameter_override("task_profile", "hazard_detection");
  opts.append_parameter_override("prompt_version", "hazard-v2");
  opts.append_parameter_override("task_instruction", "Use concise bullet points.");
  node_ = std::make_shared<VlmReasonerNode>(
    std::make_unique<FakeInferenceBackend>(),
    opts);

  std::this_thread::sleep_for(100ms);

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = helper->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);

  std::this_thread::sleep_for(50ms);
  pub->publish(*make_image(rclcpp::Time(500, 0, RCL_ROS_TIME)));

  const bool ok = spin_until(node_, helper, [&] {return received.load();});
  ASSERT_TRUE(ok) << "No result message received within timeout";
  EXPECT_EQ(last_msg.task_profile, "hazard_detection");
  EXPECT_EQ(last_msg.prompt_version, "hazard-v2");
  EXPECT_FALSE(last_msg.prompt_config_hash.empty());
  EXPECT_NE(last_msg.prompt.find("Detect hazards in this camera frame"), std::string::npos);
}

TEST_F(NodeTest, TrackedObservationModePublishesTrackedMetadata)
{
  std::atomic<bool> received{false};
  ResultMsg last_msg;

  auto helper = std::make_shared<rclcpp::Node>("_test_helper_tracked");
  auto sub = helper->create_subscription<ResultMsg>(
    "/vlm/result", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      last_msg = *msg;
      received = true;
    });

  rclcpp::NodeOptions opts = make_options(true);
  opts.append_parameter_override("enable_tracked_observation_input", true);
  opts.append_parameter_override("tracked_observation_topic", "/tracked_observation");
  opts.append_parameter_override("sample_period_seconds", 0.0);
  node_ = std::make_shared<VlmReasonerNode>(
    std::make_unique<FakeInferenceBackend>(),
    opts);

  std::this_thread::sleep_for(100ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = helper->create_publisher<edge_vlm_ros::msg::TrackedObservation>(
    "/tracked_observation", qos);

  std::this_thread::sleep_for(50ms);
  pub->publish(*make_tracked_observation(rclcpp::Time(700, 0, RCL_ROS_TIME), 99));

  ASSERT_TRUE(spin_until(node_, helper, [&] {return received.load();}));
  EXPECT_TRUE(last_msg.success);
  EXPECT_EQ(last_msg.detector_id, "detector");
  EXPECT_EQ(last_msg.tracker_id, "tracker");
  EXPECT_EQ(last_msg.tracked_object_count, 1u);
  EXPECT_EQ(last_msg.source_sequence, 99u);
  EXPECT_NE(last_msg.tracker_context.find("Tracked objects:"), std::string::npos);
}

TEST_F(NodeTest, TrackedObservationDuplicateSequenceIsSuppressedWhilePending)
{
  auto helper = std::make_shared<rclcpp::Node>("_test_helper_tracked_duplicates");
  std::vector<uint64_t> sequences;
  std::mutex seq_mutex;
  std::atomic<int> received{0};
  auto sub = helper->create_subscription<ResultMsg>(
    "/vlm/result", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(seq_mutex);
      sequences.push_back(msg->source_sequence);
      ++received;
    });

  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("enable_tracked_observation_input", true);
  opts.append_parameter_override("tracked_observation_topic", "/tracked_observation");
  opts.append_parameter_override("sample_period_seconds", 0.0);
  opts.append_parameter_override("publish_results", true);
  node_ = std::make_shared<VlmReasonerNode>(
    std::make_unique<SlowFakeInferenceBackend>(150ms), opts);

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = helper->create_publisher<edge_vlm_ros::msg::TrackedObservation>(
    "/tracked_observation", qos);

  std::this_thread::sleep_for(100ms);
  pub->publish(*make_tracked_observation(rclcpp::Time(800, 0, RCL_ROS_TIME), 123));
  pub->publish(*make_tracked_observation(rclcpp::Time(801, 0, RCL_ROS_TIME), 123));
  pub->publish(*make_tracked_observation(rclcpp::Time(802, 0, RCL_ROS_TIME), 124));

  ASSERT_TRUE(spin_until(node_, helper, [&] {return received.load() >= 2;}, 4s));
  std::lock_guard<std::mutex> lock(seq_mutex);
  ASSERT_EQ(sequences.size(), 2u);
  EXPECT_EQ(sequences[0], 123u);
  EXPECT_EQ(sequences[1], 124u);
}

TEST_F(NodeTest, TrackedObservationMinIntervalKeepsNewestPendingObservation)
{
  std::mutex seq_mutex;
  std::vector<uint64_t> sequences;
  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest &) {
      InferenceResponse resp;
      resp.success = true;
      resp.text = "ok";
      resp.inference_seconds = 0.001;
      return resp;
    });

  std::atomic<int> received{0};
  auto helper = std::make_shared<rclcpp::Node>("_test_helper_tracked_interval");
  auto sub = helper->create_subscription<ResultMsg>(
    "/vlm/result", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(seq_mutex);
      sequences.push_back(msg->source_sequence);
      ++received;
    });

  rclcpp::NodeOptions opts = make_options(true);
  opts.append_parameter_override("enable_tracked_observation_input", true);
  opts.append_parameter_override("tracked_observation_topic", "/tracked_observation");
  opts.append_parameter_override("sample_period_seconds", 0.0);
  opts.append_parameter_override("min_vlm_interval_seconds", 0.2);
  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), opts);

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = helper->create_publisher<edge_vlm_ros::msg::TrackedObservation>(
    "/tracked_observation", qos);

  std::this_thread::sleep_for(100ms);
  pub->publish(*make_tracked_observation(rclcpp::Time(900, 0, RCL_ROS_TIME), 200));
  ASSERT_TRUE(spin_until(node_, helper, [&] {return received.load() >= 1;}));

  pub->publish(*make_tracked_observation(rclcpp::Time(901, 0, RCL_ROS_TIME), 201));
  pub->publish(*make_tracked_observation(rclcpp::Time(902, 0, RCL_ROS_TIME), 202));

  ASSERT_TRUE(spin_until(node_, helper, [&] {return received.load() >= 2;}, 4s));
  std::lock_guard<std::mutex> lock(seq_mutex);
  ASSERT_EQ(sequences.size(), 2u);
  EXPECT_EQ(sequences[0], 200u);
  EXPECT_EQ(sequences[1], 202u);
}

/// Unknown variables in prompt templates fail validation during startup.
TEST_F(NodeTest, InvalidTemplateVariableRejected)
{
  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("task_profile", "scene_description");
  opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Describe scene with {not_a_valid_variable}");

  EXPECT_THROW(
    std::make_shared<VlmReasonerNode>(
      std::make_unique<FakeInferenceBackend>(), opts),
    std::exception);
}

/// Unsupported separate instruction delivery mode is rejected during startup.
TEST_F(NodeTest, UnsupportedSeparateInstructionModeRejected)
{
  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("instruction_delivery_mode", "separate");
  EXPECT_THROW(
    std::make_shared<VlmReasonerNode>(
      std::make_unique<FakeInferenceBackend>(), opts),
    std::exception);
}

/// Prompt rendering does not duplicate explicitly templated instructions.
TEST_F(NodeTest, InstructionVariablesAreNotDuplicated)
{
  std::mutex prompts_mutex;
  std::vector<std::string> captured_prompts;
  std::atomic<int> calls{0};

  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest & req) {
      {
        std::lock_guard<std::mutex> lock(prompts_mutex);
        captured_prompts.push_back(req.prompt);
      }
      ++calls;
      InferenceResponse resp;
      resp.success = true;
      resp.text = "ok";
      resp.inference_seconds = 0.001;
      return resp;
    });

  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "System: {system_instruction}\nTask: {task_instruction}\nDescribe frame.");
  opts.append_parameter_override("task_profile", "scene_description");
  opts.append_parameter_override("system_instruction", "Conservative output.");
  opts.append_parameter_override("task_instruction", "Bulleted response.");
  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), opts);

  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(550, 0, RCL_ROS_TIME)));
  const bool ok = spin_until(node_, nullptr, [&] {return calls.load() >= 1;});
  ASSERT_TRUE(ok) << "Expected one inference call";

  std::lock_guard<std::mutex> lock(prompts_mutex);
  ASSERT_EQ(captured_prompts.size(), 1u);
  const std::string & prompt = captured_prompts[0];
  const auto system_first = prompt.find("Conservative output.");
  ASSERT_NE(system_first, std::string::npos);
  EXPECT_EQ(prompt.find("Conservative output.", system_first + 1), std::string::npos);
  const auto task_first = prompt.find("Bulleted response.");
  ASSERT_NE(task_first, std::string::npos);
  EXPECT_EQ(prompt.find("Bulleted response.", task_first + 1), std::string::npos);
}

/// Observation-history injection stays bounded by observation_history_max_entries.
TEST_F(NodeTest, PromptHistoryIsBoundedByEntries)
{
  std::mutex prompts_mutex;
  std::vector<std::string> captured_prompts;
  std::atomic<int> calls{0};

  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest & req) {
      {
        std::lock_guard<std::mutex> lock(prompts_mutex);
        captured_prompts.push_back(req.prompt);
      }
      const int index = ++calls;
      InferenceResponse resp;
      resp.success = true;
      resp.text = "response " + std::to_string(index);
      resp.inference_seconds = 0.001;
      return resp;
    });

  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("observation_history_max_entries", 1);
  opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Current frame. Prior context:\n{context}");
  opts.append_parameter_override("task_profile", "scene_description");

  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), opts);

  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(600, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 1;}));
  pub->publish(*make_image(rclcpp::Time(601, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 2;}));
  pub->publish(*make_image(rclcpp::Time(602, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 3;}));

  std::lock_guard<std::mutex> lock(prompts_mutex);
  ASSERT_GE(captured_prompts.size(), 3u);
  EXPECT_EQ(captured_prompts[0].find("response 1"), std::string::npos);
  EXPECT_NE(captured_prompts[1].find("response 1"), std::string::npos);
  EXPECT_NE(captured_prompts[2].find("response 2"), std::string::npos);
  EXPECT_EQ(captured_prompts[2].find("response 1"), std::string::npos);
}

/// Observation history is reset when a failed inference occurs under on_error policy.
TEST_F(NodeTest, PromptHistoryResetsOnError)
{
  std::mutex prompts_mutex;
  std::vector<std::string> captured_prompts;
  std::atomic<int> calls{0};

  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest & req) {
      {
        std::lock_guard<std::mutex> lock(prompts_mutex);
        captured_prompts.push_back(req.prompt);
      }
      const int index = ++calls;
      InferenceResponse resp;
      resp.inference_seconds = 0.001;
      if (index == 2) {
        resp.success = false;
        resp.error = "forced failure";
        return resp;
      }
      resp.success = true;
      resp.text = "response " + std::to_string(index);
      return resp;
    });

  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("observation_history_max_entries", 2);
  opts.append_parameter_override("observation_history_reset_policy", "on_error");
  opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Current frame. Prior context:\n{context}");
  opts.append_parameter_override("task_profile", "scene_description");

  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), opts);
  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(610, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 1;}));
  pub->publish(*make_image(rclcpp::Time(611, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 2;}));
  pub->publish(*make_image(rclcpp::Time(612, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 3;}));

  std::lock_guard<std::mutex> lock(prompts_mutex);
  ASSERT_GE(captured_prompts.size(), 3u);
  EXPECT_NE(captured_prompts[1].find("response 1"), std::string::npos);
  EXPECT_EQ(captured_prompts[2].find("response 1"), std::string::npos);
}

/// every_n_requests reset policy clears history exactly at request boundaries.
TEST_F(NodeTest, PromptHistoryEveryNRequestsBoundary)
{
  std::mutex prompts_mutex;
  std::vector<std::string> captured_prompts;
  std::atomic<int> calls{0};

  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest & req) {
      {
        std::lock_guard<std::mutex> lock(prompts_mutex);
        captured_prompts.push_back(req.prompt);
      }
      const int index = ++calls;
      InferenceResponse resp;
      resp.success = true;
      resp.text = "response " + std::to_string(index);
      resp.inference_seconds = 0.001;
      return resp;
    });

  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("observation_history_max_entries", 4);
  opts.append_parameter_override("observation_history_reset_policy", "every_n_requests");
  opts.append_parameter_override("observation_history_reset_interval_requests", 2);
  opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Current frame. Prior context:\n{context}");
  opts.append_parameter_override("task_profile", "scene_description");

  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), opts);
  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(620, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 1;}));
  pub->publish(*make_image(rclcpp::Time(621, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 2;}));
  pub->publish(*make_image(rclcpp::Time(622, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 3;}));

  std::lock_guard<std::mutex> lock(prompts_mutex);
  ASSERT_GE(captured_prompts.size(), 3u);
  EXPECT_NE(captured_prompts[1].find("response 1"), std::string::npos);
  EXPECT_EQ(captured_prompts[2].find("response 1"), std::string::npos);
  EXPECT_EQ(captured_prompts[2].find("response 2"), std::string::npos);
}

/// History can be enabled while unused when template omits {context}.
TEST_F(NodeTest, HistoryEnabledWithoutContextVariableDoesNotInjectHistory)
{
  std::mutex prompts_mutex;
  std::vector<std::string> captured_prompts;
  std::atomic<int> calls{0};

  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest & req) {
      {
        std::lock_guard<std::mutex> lock(prompts_mutex);
        captured_prompts.push_back(req.prompt);
      }
      const int index = ++calls;
      InferenceResponse resp;
      resp.success = true;
      resp.text = "response " + std::to_string(index);
      resp.inference_seconds = 0.001;
      return resp;
    });

  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("observation_history_max_entries", 2);
  opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Describe the current frame only.");
  opts.append_parameter_override("task_profile", "scene_description");

  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), opts);
  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(630, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 1;}));
  pub->publish(*make_image(rclcpp::Time(631, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 2;}));

  std::lock_guard<std::mutex> lock(prompts_mutex);
  ASSERT_GE(captured_prompts.size(), 2u);
  EXPECT_EQ(captured_prompts[1].find("response 1"), std::string::npos);
}

/// Observation history is bounded by total character size.
TEST_F(NodeTest, PromptHistoryIsBoundedByCharacterSize)
{
  std::mutex prompts_mutex;
  std::vector<std::string> captured_prompts;
  std::atomic<int> calls{0};

  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest & req) {
      {
        std::lock_guard<std::mutex> lock(prompts_mutex);
        captured_prompts.push_back(req.prompt);
      }
      const int index = ++calls;
      InferenceResponse resp;
      resp.success = true;
      if (index == 1) {
        resp.text = "AAAAAAAAAA";
      } else if (index == 2) {
        resp.text = "BBBBBBBBBB";
      } else if (index == 3) {
        resp.text = "CCCCCCCCCC";
      } else {
        resp.text = "DD";
      }
      resp.inference_seconds = 0.001;
      return resp;
    });

  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("observation_history_max_entries", 4);
  opts.append_parameter_override("observation_history_max_chars", 20);
  opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Current frame. Prior context:\n{context}");
  opts.append_parameter_override("task_profile", "scene_description");

  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), opts);
  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(640, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 1;}));
  pub->publish(*make_image(rclcpp::Time(641, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 2;}));
  pub->publish(*make_image(rclcpp::Time(642, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 3;}));
  pub->publish(*make_image(rclcpp::Time(643, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 4;}));

  std::lock_guard<std::mutex> lock(prompts_mutex);
  ASSERT_GE(captured_prompts.size(), 4u);
  EXPECT_EQ(captured_prompts[3].find("AAAAAAAAAA"), std::string::npos);
  EXPECT_NE(captured_prompts[3].find("BBBBBBBBBB"), std::string::npos);
  EXPECT_NE(captured_prompts[3].find("CCCCCCCCCC"), std::string::npos);
}

/// Structured history accounts for both user and assistant text when enforcing the limit.
TEST_F(NodeTest, StructuredPromptHistoryCountsBothSidesOfEachTurn)
{
  std::mutex histories_mutex;
  std::vector<std::vector<HistoryEntry>> captured_histories;
  std::atomic<int> calls{0};

  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest & req) {
      {
        std::lock_guard<std::mutex> lock(histories_mutex);
        captured_histories.push_back(req.history);
      }
      const int index = ++calls;
      InferenceResponse resp;
      resp.success = true;
      if (index == 1) {
        resp.text = "AAAAAAAAAA";
      } else if (index == 2) {
        resp.text = "BBBBBBBBBB";
      } else if (index == 3) {
        resp.text = "CCCCCCCCCC";
      } else {
        resp.text = "DD";
      }
      resp.inference_seconds = 0.001;
      return resp;
    });

  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("instruction_delivery_mode", "structured");
  opts.append_parameter_override("observation_history_max_entries", 4);
  // The rendered user text is one character and each response is ten characters.
  // Two complete structured turns fit exactly; a third evicts the oldest.
  opts.append_parameter_override("observation_history_max_chars", 22);
  opts.append_parameter_override("task_profiles.scene_description.template", "x");
  opts.append_parameter_override("task_profile", "scene_description");

  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), opts);
  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(650, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 1;}));
  pub->publish(*make_image(rclcpp::Time(651, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 2;}));
  pub->publish(*make_image(rclcpp::Time(652, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 3;}));
  pub->publish(*make_image(rclcpp::Time(653, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 4;}));

  std::lock_guard<std::mutex> lock(histories_mutex);
  ASSERT_GE(captured_histories.size(), 4u);
  ASSERT_EQ(captured_histories[3].size(), 2u);
  EXPECT_EQ(captured_histories[3][0].user_text, "x");
  EXPECT_EQ(captured_histories[3][0].asst_text, "BBBBBBBBBB");
  EXPECT_EQ(captured_histories[3][1].user_text, "x");
  EXPECT_EQ(captured_histories[3][1].asst_text, "CCCCCCCCCC");
}

/// Malformed templates fail validation and doubled braces render literal braces.
TEST_F(NodeTest, TemplateBraceValidationAndLiteralBraces)
{
  rclcpp::NodeOptions bad_opts = make_options(false);
  bad_opts.append_parameter_override("task_profile", "scene_description");
  bad_opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Malformed template }");
  EXPECT_THROW(
    std::make_shared<VlmReasonerNode>(
      std::make_unique<FakeInferenceBackend>(), bad_opts),
    std::exception);

  std::mutex prompts_mutex;
  std::vector<std::string> captured_prompts;
  std::atomic<int> calls{0};
  auto backend = std::make_unique<FakeInferenceBackend>(
    [&](const InferenceRequest & req) {
      {
        std::lock_guard<std::mutex> lock(prompts_mutex);
        captured_prompts.push_back(req.prompt);
      }
      ++calls;
      InferenceResponse resp;
      resp.success = true;
      resp.text = "ok";
      resp.inference_seconds = 0.001;
      return resp;
    });

  rclcpp::NodeOptions ok_opts = make_options(false);
  ok_opts.append_parameter_override("task_profile", "scene_description");
  ok_opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Literal {{brace}} and topic {source_topic}");
  node_ = std::make_shared<VlmReasonerNode>(std::move(backend), ok_opts);
  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(650, 0, RCL_ROS_TIME)));
  ASSERT_TRUE(spin_until(node_, nullptr, [&] {return calls.load() >= 1;}));

  std::lock_guard<std::mutex> lock(prompts_mutex);
  ASSERT_EQ(captured_prompts.size(), 1u);
  EXPECT_NE(captured_prompts[0].find("Literal {brace}"), std::string::npos);
  EXPECT_NE(captured_prompts[0].find("/camera/image_raw"), std::string::npos);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  auto result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
