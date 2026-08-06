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

#include "edge_vlm_ros/yolo_detection2d_adapter.hpp"

#include <stdexcept>

#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

namespace edge_vlm_ros
{

namespace
{

vision_msgs::msg::Detection2D convert_detection(const yolo_msgs::msg::Detection & input)
{
  vision_msgs::msg::Detection2D output;
  output.bbox.center.position.x = input.bbox.center.position.x;
  output.bbox.center.position.y = input.bbox.center.position.y;
  output.bbox.size_x = input.bbox.size.x;
  output.bbox.size_y = input.bbox.size.y;

  vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
  hypothesis.hypothesis.class_id = input.class_name.empty() ?
    std::to_string(input.class_id) : input.class_name;
  hypothesis.hypothesis.score = input.score;
  output.results.push_back(std::move(hypothesis));
  return output;
}

}  // namespace

vision_msgs::msg::Detection2DArray convert_yolo_detection_array(
  const yolo_msgs::msg::DetectionArray & input)
{
  vision_msgs::msg::Detection2DArray output;
  output.header = input.header;
  output.detections.reserve(input.detections.size());
  for (const auto & detection : input.detections) {
    output.detections.push_back(convert_detection(detection));
  }
  return output;
}

YoloDetection2DAdapter::YoloDetection2DAdapter(const rclcpp::NodeOptions & options)
: Node("yolo_detection2d_adapter", options)
{
  declare_parameters();
  validate_parameters();

  output_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    output_topic_, rclcpp::SensorDataQoS());
  input_sub_ = create_subscription<yolo_msgs::msg::DetectionArray>(
    input_topic_, rclcpp::SensorDataQoS(),
    [this](const yolo_msgs::msg::DetectionArray::ConstSharedPtr msg) {
      detections_callback(msg);
    });
}

void YoloDetection2DAdapter::declare_parameters()
{
  const auto desc = [](const std::string & text) {
      rcl_interfaces::msg::ParameterDescriptor d;
      d.description = text;
      return d;
    };

  declare_parameter(
    "input_topic", "/yolo/detections",
    desc("Input yolo_msgs/msg/DetectionArray topic"));
  declare_parameter(
    "output_topic", "/detections",
    desc("Output vision_msgs/msg/Detection2DArray topic"));
}

void YoloDetection2DAdapter::validate_parameters()
{
  input_topic_ = get_parameter("input_topic").as_string();
  output_topic_ = get_parameter("output_topic").as_string();
  if (input_topic_.empty() || output_topic_.empty()) {
    throw std::runtime_error("YoloDetection2DAdapter topics must be non-empty");
  }
}

void YoloDetection2DAdapter::detections_callback(
  const yolo_msgs::msg::DetectionArray::ConstSharedPtr & msg)
{
  output_pub_->publish(convert_yolo_detection_array(*msg));
}

}  // namespace edge_vlm_ros
