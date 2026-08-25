// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

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

#include "edge_vlm_ros/vlm_reasoner_node.hpp"
#include "fake_inference_backend.hpp"

using namespace std::chrono_literals;
using edge_vlm_ros::VlmReasonerNode;
using edge_vlm_ros::FakeInferenceBackend;

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

class FrameSamplingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    auto backend = std::make_unique<FakeInferenceBackend>();
    node_ = std::make_shared<VlmReasonerNode>(
      std::move(backend), node_options_no_trt());
  }

  void TearDown() override
  {
    node_.reset();
  }

  std::shared_ptr<VlmReasonerNode> node_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr make_pub()
  {
    rclcpp::QoS qos{rclcpp::KeepLast(10)};
    qos.best_effort();
    return node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  }
};

TEST_F(FrameSamplingTest, FirstFrameAlwaysSampled)
{
  auto pub = make_pub();
  std::this_thread::sleep_for(50ms);

  publish_and_spin(pub, node_, make_image(rclcpp::Time(0, 0, RCL_ROS_TIME)));

  EXPECT_GE(node_->sampled_count(), 1u);
}

TEST_F(FrameSamplingTest, FramesWithinPeriodAreNotSampled)
{
  auto pub = make_pub();
  std::this_thread::sleep_for(50ms);

  publish_and_spin(pub, node_, make_image(rclcpp::Time(0, 0, RCL_ROS_TIME)));
  uint64_t after_first = node_->sampled_count();
  EXPECT_GE(after_first, 1u) << "First frame should be sampled";

  publish_and_spin(pub, node_, make_image(rclcpp::Time(1, 0, RCL_ROS_TIME)));

  EXPECT_EQ(node_->sampled_count(), after_first)
    << "Frame within period should not increment sampled_count";
}

TEST_F(FrameSamplingTest, FramesAtPeriodAreSampled)
{
  auto pub = make_pub();
  std::this_thread::sleep_for(50ms);

  publish_and_spin(pub, node_, make_image(rclcpp::Time(0, 0, RCL_ROS_TIME)));
  EXPECT_GE(node_->sampled_count(), 1u);

  publish_and_spin(pub, node_, make_image(rclcpp::Time(2, 0, RCL_ROS_TIME)));
  EXPECT_GE(node_->sampled_count(), 2u);

  publish_and_spin(pub, node_, make_image(rclcpp::Time(4, 0, RCL_ROS_TIME)));
  EXPECT_GE(node_->sampled_count(), 3u);
}

TEST_F(FrameSamplingTest, SamplingIsDeterministicWithBagTimestamps)
{
  auto pub = make_pub();
  std::this_thread::sleep_for(50ms);

  const std::vector<int64_t> secs = {0, 1, 2, 3, 4, 5};
  for (int64_t s : secs) {
    publish_and_spin(pub, node_, make_image(rclcpp::Time(s, 0, RCL_ROS_TIME)));
  }

  EXPECT_EQ(node_->sampled_count(), 3u)
    << "With period=2s, timestamps 0,1,2,3,4,5 should yield 3 sampled frames";
}

TEST_F(FrameSamplingTest, BackwardTimestampResetsSampler)
{
  auto pub = make_pub();
  std::this_thread::sleep_for(50ms);

  publish_and_spin(pub, node_, make_image(rclcpp::Time(10, 0, RCL_ROS_TIME)));
  const uint64_t before_rewind = node_->sampled_count();
  ASSERT_GE(before_rewind, 1u);

  publish_and_spin(pub, node_, make_image(rclcpp::Time(1, 0, RCL_ROS_TIME)));
  EXPECT_EQ(node_->sampled_count(), before_rewind + 1);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  auto result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
