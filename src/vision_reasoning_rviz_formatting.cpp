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

#include "edge_vlm_ros/vision_reasoning_rviz_formatting.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace edge_vlm_ros::rviz
{

std::string format_stamp(const builtin_interfaces::msg::Time & stamp)
{
  std::ostringstream out;
  out << stamp.sec << "." << std::setfill('0') << std::setw(9) << stamp.nanosec;
  return out.str();
}

std::string format_stamp(const rclcpp::Time & stamp)
{
  const int64_t ns_per_sec = 1000000000LL;
  int64_t total_ns = stamp.nanoseconds();
  int64_t sec = total_ns / ns_per_sec;
  int64_t nanosec = total_ns % ns_per_sec;
  if (nanosec < 0) {
    --sec;
    nanosec += ns_per_sec;
  }

  std::ostringstream out;
  out << sec << "." << std::setfill('0') << std::setw(9) << nanosec;
  return out.str();
}

ResultPresentation build_result_presentation(
  const msg::VlmResult * result,
  const std::optional<rclcpp::Time> & latest_image_stamp,
  const rclcpp::Time & now,
  double stale_after_seconds)
{
  ResultPresentation presentation;
  if (!result) {
    presentation.state = ResultState::kNoResult;
    presentation.status_text = "NO RESULT";
    presentation.details_text = "Waiting for VlmResult messages.";
    return presentation;
  }

  const rclcpp::Time result_stamp(result->header.stamp, RCL_ROS_TIME);
  const double clamped_stale_after = std::max(0.0, stale_after_seconds);
  const double age_seconds = std::max(0.0, (now - result_stamp).seconds());
  const bool stale_by_age = age_seconds > clamped_stale_after;
  const bool stale_by_newer_image = latest_image_stamp.has_value() && latest_image_stamp.value() > result_stamp;
  const bool stale = stale_by_age || stale_by_newer_image;

  if (!result->success) {
    presentation.state = ResultState::kFailed;
    presentation.status_text = "FAILED";
  } else if (stale) {
    presentation.state = ResultState::kStale;
    presentation.status_text = "STALE";
  } else {
    presentation.state = ResultState::kSuccess;
    presentation.status_text = "SUCCESS";
  }

  std::ostringstream details;
  details << "result_stamp=" << format_stamp(result->header.stamp)
          << "  frame_sequence=" << result->frame_sequence
          << "  latency_s=" << std::fixed << std::setprecision(3) << result->inference_seconds
          << "  age_s=" << std::fixed << std::setprecision(3) << age_seconds;
  if (stale_by_newer_image && latest_image_stamp.has_value()) {
    details << "  newer_image_stamp=" << format_stamp(latest_image_stamp.value());
  }
  presentation.details_text = details.str();
  return presentation;
}

}  // namespace edge_vlm_ros::rviz
