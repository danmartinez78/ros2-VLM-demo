// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

#include "edge_vlm_ros/tensorrt_edge_llm_backend.hpp"

#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

int main(int argc, char ** argv)
{
  if (argc != 5) {
    std::cerr
      << "Usage: " << argv[0]
      << " <llm-engine-dir> <multimodal-engine-dir> <plugin-path> <image>\n";
    return 2;
  }

  const char * ros_mode = std::getenv("EDGE_VLM_SMOKE_ROS_MODE");
  const bool use_ros = ros_mode != nullptr;
  if (use_ros) {
    std::cout << "ROS mode: " << ros_mode << '\n';
    rclcpp::init(argc, argv);
    if (std::string(ros_mode) == "node") {
      auto node = std::make_shared<rclcpp::Node>("edge_vlm_smoke_parameter_node");
      std::cout << "Temporary ROS node constructed\n";
      node.reset();
      std::cout << "Temporary ROS node destroyed\n";
    } else if (std::string(ros_mode) != "init") {
      std::cerr << "EDGE_VLM_SMOKE_ROS_MODE must be 'init' or 'node'\n";
      rclcpp::shutdown();
      return 2;
    }
  } else {
    std::cout << "ROS mode: disabled\n";
  }

  auto run_backend = [&]() -> int {
    try {
      edge_vlm_ros::TensorRTEdgeLLMConfig config;
      config.llm_engine_dir = argv[1];
      config.multimodal_engine_dir = argv[2];
      config.edge_llm_plugin_path = argv[3];
      config.jpeg_quality = 90;

      cv::Mat image = cv::imread(argv[4], cv::IMREAD_COLOR);
      if (image.empty()) {
        std::cerr << "Could not read image: " << argv[4] << '\n';
        return 3;
      }

      std::cout << "Image: " << image.cols << "x" << image.rows << '\n';
      std::cout << "Initializing direct-linked backend...\n";

      edge_vlm_ros::TensorRTEdgeLLMBackend backend(config);
      backend.initialize();

      edge_vlm_ros::InferenceRequest request;
      request.image = image;
      request.prompt = "Describe what you see in this image.";
      request.max_generate_length = 64;
      request.temperature = 1.0F;
      request.top_p = 1.0F;
      request.top_k = 50;

      std::cout << "Backend initialized. Starting inference...\n";
      auto const response = backend.infer(request);

      if (!response.success) {
        std::cerr << "Inference failed: " << response.error << '\n';
        return 4;
      }

      std::cout << "Inference time: " << response.inference_seconds << " seconds\n";
      std::cout << "Response: " << response.text << '\n';
      return 0;
    } catch (std::exception const & error) {
      std::cerr << "Exception: " << error.what() << '\n';
      return 5;
    }
  };

  int result = 5;
  if (std::getenv("EDGE_VLM_SMOKE_THREADED") == nullptr) {
    std::cout << "Execution mode: main thread\n";
    result = run_backend();
  } else {
    std::cout << "Execution mode: std::thread worker\n";
    std::thread worker([&]() {result = run_backend();});
    worker.join();
  }

  if (use_ros) {
    rclcpp::shutdown();
  }
  return result;
}
