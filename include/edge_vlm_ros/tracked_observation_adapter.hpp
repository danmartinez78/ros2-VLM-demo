// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "edge_vlm_ros/iou_tracker.hpp"
#include "edge_vlm_ros/msg/tracked_observation.hpp"

namespace edge_vlm_ros
{

class TrackedObservationAdapter : public rclcpp::Node
{
public:
  explicit TrackedObservationAdapter(const rclcpp::NodeOptions & options = rclcpp::NodeOptions{});

private:
  using DetectionArray = vision_msgs::msg::Detection2DArray;

  void declare_parameters();
  void validate_parameters();
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & image_msg);
  void detections_callback(const DetectionArray::ConstSharedPtr & detections_msg);
  void try_publish_latest_match();
  msg::TrackedObservation build_observation(
    const sensor_msgs::msg::Image & image_msg,
    const DetectionArray & detections_msg,
    const rclcpp::Time & completed_at);

  std::string image_topic_;
  std::string detections_topic_;
  std::string tracked_observation_topic_;
  std::string detector_id_;
  std::string tracker_id_;
  uint64_t mismatch_drop_count_{0};
  uint64_t stale_drop_count_{0};
  float tracker_min_iou_{0.3f};
  int tracker_max_coast_age_{1};
  bool tracker_class_aware_{true};

  IouTracker tracker_;
  uint64_t next_source_sequence_{1};
  uint64_t published_count_{0};

  std::optional<sensor_msgs::msg::Image> latest_image_;
  std::optional<DetectionArray> latest_detections_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<DetectionArray>::SharedPtr detections_sub_;
  rclcpp::Publisher<msg::TrackedObservation>::SharedPtr tracked_observation_pub_;
};

}  // namespace edge_vlm_ros
