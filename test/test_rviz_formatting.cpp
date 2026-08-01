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

#include <gtest/gtest.h>

#include <optional>

#include <rclcpp/rclcpp.hpp>

#include "edge_vlm_ros/msg/vlm_result.hpp"
#include "edge_vlm_ros/vision_reasoning_rviz_formatting.hpp"

using edge_vlm_ros::msg::VlmResult;
using edge_vlm_ros::rviz::ResultState;

TEST(RvizFormatting, NoResultShowsWaiting)
{
  auto presentation = edge_vlm_ros::rviz::build_result_presentation(
    nullptr,
    std::nullopt,
    rclcpp::Time(10, 0, RCL_ROS_TIME),
    2.0);

  EXPECT_EQ(presentation.state, ResultState::kNoResult);
  EXPECT_EQ(presentation.status_text, "NO RESULT");
}

TEST(RvizFormatting, RecentSuccessfulResultIsFresh)
{
  VlmResult result;
  result.header.stamp = rclcpp::Time(9, 500000000, RCL_ROS_TIME);
  result.success = true;
  result.frame_sequence = 5;
  result.inference_seconds = 0.75;

  auto presentation = edge_vlm_ros::rviz::build_result_presentation(
    &result,
    std::optional<rclcpp::Time>(rclcpp::Time(9, 500000000, RCL_ROS_TIME)),
    rclcpp::Time(10, 0, RCL_ROS_TIME),
    2.0);

  EXPECT_EQ(presentation.state, ResultState::kSuccess);
  EXPECT_EQ(presentation.status_text, "SUCCESS");
}

TEST(RvizFormatting, NewerImageMarksResultStale)
{
  VlmResult result;
  result.header.stamp = rclcpp::Time(8, 0, RCL_ROS_TIME);
  result.success = true;

  auto presentation = edge_vlm_ros::rviz::build_result_presentation(
    &result,
    std::optional<rclcpp::Time>(rclcpp::Time(9, 0, RCL_ROS_TIME)),
    rclcpp::Time(9, 0, RCL_ROS_TIME),
    10.0);

  EXPECT_EQ(presentation.state, ResultState::kStale);
  EXPECT_EQ(presentation.status_text, "STALE");
}

TEST(RvizFormatting, FailedResultOverridesFreshness)
{
  VlmResult result;
  result.header.stamp = rclcpp::Time(9, 0, RCL_ROS_TIME);
  result.success = false;
  result.error = "backend failed";

  auto presentation = edge_vlm_ros::rviz::build_result_presentation(
    &result,
    std::optional<rclcpp::Time>(rclcpp::Time(9, 500000000, RCL_ROS_TIME)),
    rclcpp::Time(10, 0, RCL_ROS_TIME),
    0.1);

  EXPECT_EQ(presentation.state, ResultState::kFailed);
  EXPECT_EQ(presentation.status_text, "FAILED");
}
