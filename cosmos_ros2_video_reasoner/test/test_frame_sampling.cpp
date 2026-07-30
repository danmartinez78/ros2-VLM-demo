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

/// Tests for timestamp-based frame sampling logic.
///
/// These tests do NOT require TensorRT, CUDA, or any NVIDIA hardware.

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "cosmos_ros2_video_reasoner/cosmos_reasoner_node.hpp"
#include "fake_inference_backend.hpp"

using namespace std::chrono_literals;
using cosmos_ros2_video_reasoner::CosmosReasonerNode;
using cosmos_ros2_video_reasoner::FakeInferenceBackend;

// ─── helpers ─────────────────────────────────────────────────────────────────

static sensor_msgs::msg::Image::SharedPtr make_image(rclcpp::Time stamp)
{
  auto img = std::make_shared<sensor_msgs::msg::Image>();
  img->header.stamp = stamp;
  img->encoding = "bgr8";
  img->width = 4;
  img->height = 4;
  img->step = 4 * 3;
  img->data.assign(4 * 4 * 3, 128u);
  return img;
}

static rclcpp::NodeOptions node_options_no_trt(double period = 2.0)
{
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("llm_engine_dir", "/nonexistent");
  opts.append_parameter_override("multimodal_engine_dir", "/nonexistent");
  opts.append_parameter_override("edge_llm_plugin_path", "/nonexistent");
  opts.append_parameter_override("sample_period_seconds", period);
  opts.append_parameter_override("publish_results", false);
  return opts;
}

// Publish messages and spin until the callback has processed them.
static void publish_and_spin(
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub,
  rclcpp::Node::SharedPtr node,
  sensor_msgs::msg::Image::SharedPtr img,
  std::chrono::milliseconds timeout = 200ms)
{
  pub->publish(*img);
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(5ms);
  }
}

// ─── Test fixture ─────────────────────────────────────────────────────────────

class FrameSamplingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    auto backend = std::make_unique<FakeInferenceBackend>();
    node_ = std::make_shared<CosmosReasonerNode>(
      std::move(backend), node_options_no_trt());
  }

  void TearDown() override
  {
    node_.reset();
  }

  std::shared_ptr<CosmosReasonerNode> node_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr make_pub()
  {
    // Use best_effort to match the subscription QoS
    rclcpp::QoS qos{rclcpp::KeepLast(10)};
    qos.best_effort();
    return node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  }
};

// ─── Tests ───────────────────────────────────────────────────────────────────

/// The first frame is always sampled regardless of the period.
TEST_F(FrameSamplingTest, FirstFrameAlwaysSampled)
{
  auto pub = make_pub();
  // Small sleep to let DDS discover the subscription
  std::this_thread::sleep_for(50ms);

  publish_and_spin(pub, node_, make_image(rclcpp::Time(0, 0, RCL_ROS_TIME)));

  EXPECT_GE(node_->sampled_count(), 1u);
}

/// Frames within the sample period are NOT sampled.
TEST_F(FrameSamplingTest, FramesWithinPeriodAreNotSampled)
{
  auto pub = make_pub();
  std::this_thread::sleep_for(50ms);

  // t=0 → sampled
  publish_and_spin(pub, node_, make_image(rclcpp::Time(0, 0, RCL_ROS_TIME)));
  uint64_t after_first = node_->sampled_count();
  EXPECT_GE(after_first, 1u) << "First frame should be sampled";

  // t=1 s < 2 s period → NOT sampled
  publish_and_spin(pub, node_, make_image(rclcpp::Time(1, 0, RCL_ROS_TIME)));

  EXPECT_EQ(node_->sampled_count(), after_first)
    << "Frame within period should not increment sampled_count";
}

/// Frames at or beyond the sample period ARE sampled.
TEST_F(FrameSamplingTest, FramesAtPeriodAreSampled)
{
  auto pub = make_pub();
  std::this_thread::sleep_for(50ms);

  // t=0 → sampled
  publish_and_spin(pub, node_, make_image(rclcpp::Time(0, 0, RCL_ROS_TIME)));
  EXPECT_GE(node_->sampled_count(), 1u);

  // t=2 s == 2 s period → sampled
  publish_and_spin(pub, node_, make_image(rclcpp::Time(2, 0, RCL_ROS_TIME)));
  EXPECT_GE(node_->sampled_count(), 2u);

  // t=4 s → sampled
  publish_and_spin(pub, node_, make_image(rclcpp::Time(4, 0, RCL_ROS_TIME)));
  EXPECT_GE(node_->sampled_count(), 3u);
}

/// Sampling is deterministic with bag timestamps.
TEST_F(FrameSamplingTest, SamplingIsDeterministicWithBagTimestamps)
{
  auto pub = make_pub();
  std::this_thread::sleep_for(50ms);

  const std::vector<int64_t> secs = {0, 1, 2, 3, 4, 5};
  // With period=2, expected sampled: t=0, t=2, t=4 → 3 frames
  for (int64_t s : secs) {
    publish_and_spin(pub, node_, make_image(rclcpp::Time(s, 0, RCL_ROS_TIME)));
  }

  EXPECT_EQ(node_->sampled_count(), 3u)
    << "With period=2s, timestamps 0,1,2,3,4,5 should yield 3 sampled frames";
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  auto result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
