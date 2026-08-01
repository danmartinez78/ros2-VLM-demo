// Copyright 2025 edge_vlm_ros contributors
#include "edge_vlm_ros/ipc_inference_backend.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
void usage(char const * program)
{
  std::cerr
    << "Usage: " << program
    << " --socket PATH --image PATH [--prompt TEXT]"
    << " [--max-generate-length N] [--temperature F] [--top-p F] [--top-k N]\n";
}

std::string require_value(int argc, char ** argv, int & index)
{
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
}
}  // namespace

int main(int argc, char ** argv)
{
  edge_vlm_ros::IpcInferenceConfig config;
  edge_vlm_ros::InferenceRequest request;
  request.prompt = "Describe the scene in this image.";
  request.max_generate_length = 64;
  request.temperature = 0.2F;
  request.top_p = 0.9F;
  request.top_k = 20;

  std::string image_path;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--help" || option == "-h") {
        usage(argv[0]);
        return 0;
      } else if (option == "--socket") {
        config.socket_path = require_value(argc, argv, index);
      } else if (option == "--image") {
        image_path = require_value(argc, argv, index);
      } else if (option == "--prompt") {
        request.prompt = require_value(argc, argv, index);
      } else if (option == "--max-generate-length") {
        request.max_generate_length = std::stoi(require_value(argc, argv, index));
      } else if (option == "--temperature") {
        request.temperature = std::stof(require_value(argc, argv, index));
      } else if (option == "--top-p") {
        request.top_p = std::stof(require_value(argc, argv, index));
      } else if (option == "--top-k") {
        request.top_k = std::stoi(require_value(argc, argv, index));
      } else {
        throw std::runtime_error("unknown option: " + option);
      }
    }

    if (image_path.empty()) {
      throw std::runtime_error("--image is required");
    }
    if (config.socket_path.empty()) {
      throw std::runtime_error("--socket must not be empty");
    }
    if (request.prompt.empty()) {
      throw std::runtime_error("--prompt must not be empty");
    }
    if (request.max_generate_length <= 0) {
      throw std::runtime_error("--max-generate-length must be positive");
    }

    request.image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (request.image.empty()) {
      throw std::runtime_error("failed to decode image: " + image_path);
    }

    edge_vlm_ros::IpcInferenceBackend backend(config);
    backend.initialize();
    const auto response = backend.infer(request);
    if (!response.success) {
      std::cerr << "Inference failed: " << response.error << '\n';
      return 1;
    }

    std::cout << response.text << '\n';
    std::cerr << "Inference time: " << response.inference_seconds << " seconds\n";
    return 0;
  } catch (std::exception const & error) {
    std::cerr << "edge_vlm_cli: " << error.what() << '\n';
    usage(argv[0]);
    return 2;
  }
}
