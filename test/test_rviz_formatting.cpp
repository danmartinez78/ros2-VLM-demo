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

#include <gtest/gtest.h>

#include <optional>

#include <rclcpp/rclcpp.hpp>

#include "cosmos_ros2_video_reasoner/msg/vision_reasoning_result.hpp"
#include "cosmos_ros2_video_reasoner/vision_reasoning_rviz_formatting.hpp"

using cosmos_ros2_video_reasoner::msg::VisionReasoningResult;
using cosmos_ros2_video_reasoner::rviz::ResultState;

TEST(RvizFormatting, NoResultShowsWaiting)
{
  auto presentation = cosmos_ros2_video_reasoner::rviz::build_result_presentation(
    nullptr,
    std::nullopt,
    rclcpp::Time(10, 0, RCL_ROS_TIME),
    2.0);

  EXPECT_EQ(presentation.state, ResultState::kNoResult);
  EXPECT_EQ(presentation.status_text, "NO RESULT");
}

TEST(RvizFormatting, RecentSuccessfulResultIsFresh)
{
  VisionReasoningResult result;
  result.header.stamp = rclcpp::Time(9, 500000000, RCL_ROS_TIME);
  result.success = true;
  result.frame_sequence = 5;
  result.inference_seconds = 0.75;

  auto presentation = cosmos_ros2_video_reasoner::rviz::build_result_presentation(
    &result,
    std::optional<rclcpp::Time>(rclcpp::Time(9, 500000000, RCL_ROS_TIME)),
    rclcpp::Time(10, 0, RCL_ROS_TIME),
    2.0);

  EXPECT_EQ(presentation.state, ResultState::kSuccess);
  EXPECT_EQ(presentation.status_text, "SUCCESS");
}

TEST(RvizFormatting, NewerImageMarksResultStale)
{
  VisionReasoningResult result;
  result.header.stamp = rclcpp::Time(8, 0, RCL_ROS_TIME);
  result.success = true;

  auto presentation = cosmos_ros2_video_reasoner::rviz::build_result_presentation(
    &result,
    std::optional<rclcpp::Time>(rclcpp::Time(9, 0, RCL_ROS_TIME)),
    rclcpp::Time(9, 0, RCL_ROS_TIME),
    10.0);

  EXPECT_EQ(presentation.state, ResultState::kStale);
  EXPECT_EQ(presentation.status_text, "STALE");
}

TEST(RvizFormatting, FailedResultOverridesFreshness)
{
  VisionReasoningResult result;
  result.header.stamp = rclcpp::Time(9, 0, RCL_ROS_TIME);
  result.success = false;
  result.error = "backend failed";

  auto presentation = cosmos_ros2_video_reasoner::rviz::build_result_presentation(
    &result,
    std::optional<rclcpp::Time>(rclcpp::Time(9, 500000000, RCL_ROS_TIME)),
    rclcpp::Time(10, 0, RCL_ROS_TIME),
    0.1);

  EXPECT_EQ(presentation.state, ResultState::kFailed);
  EXPECT_EQ(presentation.status_text, "FAILED");
}
