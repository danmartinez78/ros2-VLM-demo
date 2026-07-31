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

#pragma once

#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "cosmos_ros2_video_reasoner/msg/vision_reasoning_result.hpp"

namespace cosmos_ros2_video_reasoner::rviz
{

enum class ResultState
{
  kNoResult,
  kSuccess,
  kFailed,
  kStale
};

struct ResultPresentation
{
  ResultState state{ResultState::kNoResult};
  std::string status_text;
  std::string details_text;
};

std::string format_stamp(const builtin_interfaces::msg::Time & stamp);

ResultPresentation build_result_presentation(
  const msg::VisionReasoningResult * result,
  const std::optional<rclcpp::Time> & latest_image_stamp,
  const rclcpp::Time & now,
  double stale_after_seconds);

}  // namespace cosmos_ros2_video_reasoner::rviz
