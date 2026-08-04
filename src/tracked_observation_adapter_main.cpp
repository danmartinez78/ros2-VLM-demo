// Copyright 2025 edge_vlm_ros contributors

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "edge_vlm_ros/tracked_observation_adapter.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<edge_vlm_ros::TrackedObservationAdapter>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
