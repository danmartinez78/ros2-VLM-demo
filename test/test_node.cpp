// Copyright 2025 cosmos_ros2_video_reasoner contributors
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

/// Integration-style tests for CosmosReasonerNode using FakeInferenceBackend.
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

#include "cosmos_ros2_video_reasoner/cosmos_reasoner_node.hpp"
#include "cosmos_ros2_video_reasoner/msg/vision_reasoning_result.hpp"
#include "fake_inference_backend.hpp"

using namespace std::chrono_literals;
using cosmos_ros2_video_reasoner::CosmosReasonerNode;
using cosmos_ros2_video_reasoner::FakeInferenceBackend;
using cosmos_ros2_video_reasoner::FailingFakeInferenceBackend;
using cosmos_ros2_video_reasoner::InferenceRequest;
using cosmos_ros2_video_reasoner::InferenceResponse;
using cosmos_ros2_video_reasoner::SlowFakeInferenceBackend;
using ResultMsg = cosmos_ros2_video_reasoner::msg::VisionReasoningResult;

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

  std::shared_ptr<CosmosReasonerNode> node_;
};

// ─── Tests ────────────────────────────────────────────────────────────────────

/// After construction the fake backend is initialized.
TEST_F(NodeTest, BackendInitialized)
{
  auto * raw_backend = new FakeInferenceBackend();
  node_ = std::make_shared<CosmosReasonerNode>(
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
    "/cosmos/reasoning", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      last_msg = *msg;
      received = true;
    });

  node_ = std::make_shared<CosmosReasonerNode>(
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
    "/cosmos/reasoning", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      last_msg = *msg;
      received = true;
    });

  node_ = std::make_shared<CosmosReasonerNode>(
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

  node_ = std::make_shared<CosmosReasonerNode>(
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
  node_ = std::make_shared<CosmosReasonerNode>(
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
  node_ = std::make_shared<CosmosReasonerNode>(
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
    std::make_shared<CosmosReasonerNode>(
      std::make_unique<FakeInferenceBackend>(), opts),
    std::exception);
}

/// Invalid temperature is rejected during construction.
TEST_F(NodeTest, InvalidTemperatureRejected)
{
  rclcpp::NodeOptions opts = make_options(false);
  opts.append_parameter_override("temperature", -0.5);

  EXPECT_THROW(
    std::make_shared<CosmosReasonerNode>(
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
    "/cosmos/reasoning", rclcpp::SystemDefaultsQoS(),
    [&](ResultMsg::SharedPtr msg) {
      last_msg = *msg;
      received = true;
    });

  rclcpp::NodeOptions opts = make_options(true);
  opts.append_parameter_override("task_profile", "hazard_detection");
  opts.append_parameter_override("prompt_version", "hazard-v2");
  opts.append_parameter_override("task_instruction", "Use concise bullet points.");
  node_ = std::make_shared<CosmosReasonerNode>(
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
  EXPECT_NE(last_msg.prompt.find("Detect hazards in this camera frame"), std::string::npos);
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
    std::make_shared<CosmosReasonerNode>(
      std::make_unique<FakeInferenceBackend>(), opts),
    std::exception);
}

/// Context retention stays bounded by context_max_entries.
TEST_F(NodeTest, ContextRetentionIsBounded)
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
  opts.append_parameter_override("context_max_entries", 1);
  opts.append_parameter_override(
    "task_profiles.scene_description.template",
    "Current frame. Prior context:\n{context}");
  opts.append_parameter_override("task_profile", "scene_description");

  node_ = std::make_shared<CosmosReasonerNode>(std::move(backend), opts);

  std::this_thread::sleep_for(50ms);
  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto pub = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  std::this_thread::sleep_for(50ms);

  pub->publish(*make_image(rclcpp::Time(600, 0, RCL_ROS_TIME)));
  pub->publish(*make_image(rclcpp::Time(601, 0, RCL_ROS_TIME)));
  pub->publish(*make_image(rclcpp::Time(602, 0, RCL_ROS_TIME)));

  const bool ok = spin_until(node_, nullptr, [&] {return calls.load() >= 3;});
  ASSERT_TRUE(ok) << "Expected three inference calls";

  std::lock_guard<std::mutex> lock(prompts_mutex);
  ASSERT_GE(captured_prompts.size(), 3u);
  EXPECT_EQ(captured_prompts[0].find("response 1"), std::string::npos);
  EXPECT_NE(captured_prompts[1].find("response 1"), std::string::npos);
  EXPECT_NE(captured_prompts[2].find("response 2"), std::string::npos);
  EXPECT_EQ(captured_prompts[2].find("response 1"), std::string::npos);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  auto result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
