// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "edge_vlm_ros/msg/vlm_result.hpp"

namespace edge_vlm_ros::rviz
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
std::string format_stamp(const rclcpp::Time & stamp);

ResultPresentation build_result_presentation(
  const msg::VlmResult * result,
  const std::optional<rclcpp::Time> & latest_image_stamp,
  const rclcpp::Time & now,
  double stale_after_seconds);

}  // namespace edge_vlm_ros::rviz
