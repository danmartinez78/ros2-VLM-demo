// Copyright 2025 edge_vlm_ros contributors
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "edge_vlm_ros/vlm_reasoner_node.hpp"
#include "edge_vlm_ros/ipc_inference_backend.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto options = rclcpp::NodeOptions{};

  auto param_node = std::make_shared<rclcpp::Node>("edge_vlm_ros_node", options);
  param_node->declare_parameter("worker_socket_path", "/tmp/edge_vlm.sock");
  param_node->declare_parameter("worker_connect_timeout_seconds", 120);
  param_node->declare_parameter("worker_request_timeout_seconds", 90);
  param_node->declare_parameter("worker_inference_deadline_seconds", 60);

  edge_vlm_ros::IpcInferenceConfig config;
  config.socket_path = param_node->get_parameter("worker_socket_path").as_string();
  config.connect_timeout_seconds = static_cast<int>(
    param_node->get_parameter("worker_connect_timeout_seconds").as_int());
  config.request_timeout_seconds = static_cast<int>(
    param_node->get_parameter("worker_request_timeout_seconds").as_int());

  const int inference_deadline_seconds = static_cast<int>(
    param_node->get_parameter("worker_inference_deadline_seconds").as_int());

  if (inference_deadline_seconds <= 0) {
    RCLCPP_FATAL(
      param_node->get_logger(),
      "worker_inference_deadline_seconds (%d) must be > 0",
      inference_deadline_seconds);
    rclcpp::shutdown();
    return 1;
  }
  if (inference_deadline_seconds >= config.request_timeout_seconds) {
    RCLCPP_FATAL(
      param_node->get_logger(),
      "worker_inference_deadline_seconds (%d) must be less than "
      "worker_request_timeout_seconds (%d). "
      "The gap lets the worker exit before the client socket timeout fires, "
      "so the client sees a clean EOF instead of a SO_RCVTIMEO error.",
      inference_deadline_seconds,
      config.request_timeout_seconds);
    rclcpp::shutdown();
    return 1;
  }

  param_node.reset();

  auto backend =
    std::make_unique<edge_vlm_ros::IpcInferenceBackend>(config);
  auto node = std::make_shared<edge_vlm_ros::VlmReasonerNode>(
    std::move(backend), options);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
