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

#pragma once

#include <memory>
#include <string>

#include <message_filters/subscriber.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/synchronizer.hpp>
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
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, DetectionArray>;

  void declare_parameters();
  void validate_parameters();
  void handle_synced_messages(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const DetectionArray::ConstSharedPtr & detections_msg);
  msg::TrackedObservation build_observation(
    const sensor_msgs::msg::Image & image_msg,
    const DetectionArray & detections_msg,
    const rclcpp::Time & completed_at);

  std::string image_topic_;
  std::string detections_topic_;
  std::string tracked_observation_topic_;
  std::string detector_id_;
  std::string tracker_id_;
  int sync_queue_size_{10};
  float tracker_min_iou_{0.3f};
  int tracker_max_coast_age_{1};
  bool tracker_class_aware_{true};

  IouTracker tracker_;
  uint64_t next_source_sequence_{1};
  uint64_t published_count_{0};

  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<DetectionArray> detections_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;
  rclcpp::Publisher<msg::TrackedObservation>::SharedPtr tracked_observation_pub_;
};

}  // namespace edge_vlm_ros
