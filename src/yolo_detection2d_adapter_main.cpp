// Copyright 2025 edge_vlm_ros contributors

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "edge_vlm_ros/yolo_detection2d_adapter.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<edge_vlm_ros::YoloDetection2DAdapter>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
