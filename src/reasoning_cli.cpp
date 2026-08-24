// Copyright 2025 edge_vlm_ros contributors
#include "edge_vlm_ros/ipc_inference_backend.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void usage(char const * program)
{
  std::cerr
    << "Usage: " << program
    << " --socket PATH --image PATH [--image PATH ...] [--prompt TEXT]"
    << " [--max-generate-length N] [--temperature F] [--top-p F] [--top-k N]\n"
    << " [--sequence-type images|temporal_images|video]"
    << " [--fps F] [--frame-timestamps-sec CSV]\n"
    << "  --image may be specified multiple times for multi-frame requests.\n";
}

std::string require_value(int argc, char ** argv, int & index)
{
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
}

edge_vlm_ros::TemporalSequenceType parse_sequence_type(const std::string & value)
{
  if (value == "images") {
    return edge_vlm_ros::TemporalSequenceType::kImages;
  }
  if (value == "temporal_images") {
    return edge_vlm_ros::TemporalSequenceType::kTemporalImages;
  }
  if (value == "video") {
    return edge_vlm_ros::TemporalSequenceType::kVideo;
  }
  throw std::runtime_error(
          "invalid --sequence-type value: " + value +
          " (expected images|temporal_images|video)");
}

std::vector<double> parse_timestamp_csv(const std::string & csv)
{
  std::vector<double> out;
  if (csv.empty()) {
    return out;
  }
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      throw std::runtime_error("invalid --frame-timestamps-sec CSV (empty item)");
    }
    out.push_back(std::stod(token));
  }
  return out;
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

  std::vector<std::string> image_paths;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--help" || option == "-h") {
        usage(argv[0]);
        return 0;
      } else if (option == "--socket") {
        config.socket_path = require_value(argc, argv, index);
      } else if (option == "--image") {
        image_paths.push_back(require_value(argc, argv, index));
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
      } else if (option == "--sequence-type") {
        request.sequence_type = parse_sequence_type(require_value(argc, argv, index));
      } else if (option == "--fps") {
        request.fps = std::stod(require_value(argc, argv, index));
      } else if (option == "--frame-timestamps-sec") {
        request.frame_timestamps_sec = parse_timestamp_csv(require_value(argc, argv, index));
      } else {
        throw std::runtime_error("unknown option: " + option);
      }
    }

    if (image_paths.empty()) {
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
    edge_vlm_ros::detail::validate_temporal_metadata(request);

    // Load primary image.
    request.image = cv::imread(image_paths[0], cv::IMREAD_COLOR);
    if (request.image.empty()) {
      throw std::runtime_error("failed to decode image: " + image_paths[0]);
    }

    // Load extra images (multi-frame support).
    for (size_t i = 1; i < image_paths.size(); ++i) {
      cv::Mat extra = cv::imread(image_paths[i], cv::IMREAD_COLOR);
      if (extra.empty()) {
        throw std::runtime_error("failed to decode image: " + image_paths[i]);
      }
      request.extra_images.push_back(std::move(extra));
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
    std::cerr << "Requested sequence type: " << response.requested_sequence_type << '\n';
    std::cerr << "Runtime temporal encoding: " << response.runtime_temporal_encoding << '\n';
    std::cerr << "Temporal fallback used: " << (response.temporal_fallback_used ? "true" : "false")
              << '\n';
    return 0;
  } catch (std::exception const & error) {
    std::cerr << "edge_vlm_cli: " << error.what() << '\n';
    usage(argv[0]);
    return 2;
  }
}
