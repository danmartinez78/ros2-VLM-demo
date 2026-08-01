// Copyright 2025 cosmos_ros2_video_reasoner contributors
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "cosmos_ros2_video_reasoner/cosmos_reasoner_node.hpp"
#include "cosmos_ros2_video_reasoner/ipc_inference_backend.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto options = rclcpp::NodeOptions{};

  auto param_node = std::make_shared<rclcpp::Node>("cosmos_reasoner", options);
  param_node->declare_parameter("worker_socket_path", "/tmp/cosmos_edge_llm.sock");
  param_node->declare_parameter("worker_connect_timeout_seconds", 120);
  param_node->declare_parameter("worker_request_timeout_seconds", 90);
  param_node->declare_parameter("worker_inference_deadline_seconds", 60);

  cosmos_ros2_video_reasoner::IpcInferenceConfig config;
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
    std::make_unique<cosmos_ros2_video_reasoner::IpcInferenceBackend>(config);
  auto node = std::make_shared<cosmos_ros2_video_reasoner::CosmosReasonerNode>(
    std::move(backend), options);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
