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

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <yolo_msgs/msg/detection_array.hpp>

namespace edge_vlm_ros
{

class YoloDetection2DAdapter : public rclcpp::Node
{
public:
  explicit YoloDetection2DAdapter(const rclcpp::NodeOptions & options = rclcpp::NodeOptions{});

private:
  void declare_parameters();
  void validate_parameters();
  void detections_callback(const yolo_msgs::msg::DetectionArray::ConstSharedPtr & msg);

  std::string input_topic_;
  std::string output_topic_;
  rclcpp::Subscription<yolo_msgs::msg::DetectionArray>::SharedPtr input_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr output_pub_;
};

vision_msgs::msg::Detection2DArray convert_yolo_detection_array(
  const yolo_msgs::msg::DetectionArray & input);

}  // namespace edge_vlm_ros
