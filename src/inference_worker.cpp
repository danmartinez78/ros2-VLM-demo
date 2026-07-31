// Copyright 2025 cosmos_ros2_video_reasoner contributors
#include "cosmos_ros2_video_reasoner/ipc_protocol.hpp"
#include "cosmos_ros2_video_reasoner/tensorrt_edge_llm_backend.hpp"

#include <opencv2/core.hpp>

#include <csignal>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace
{
volatile std::sig_atomic_t g_stop = 0;
void stop_handler(int) {g_stop = 1;}
}

int main(int argc, char ** argv)
{
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <llm-engine-dir> <multimodal-engine-dir> <plugin-path> <socket-path>"
              << " <jpeg-quality>\n";
    return 2;
  }

  const std::string socket_path = argv[4];
  int jpeg_quality = 0;
  try {
    jpeg_quality = std::stoi(argv[5]);
  } catch (std::exception const &) {
    std::cerr << "JPEG quality must be an integer in [1, 100]\n";
    return 2;
  }
  if (jpeg_quality < 1 || jpeg_quality > 100) {
    std::cerr << "JPEG quality must be in [1, 100]\n";
    return 2;
  }
  if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
    std::cerr << "Socket path is too long\n";
    return 2;
  }

  std::signal(SIGINT, stop_handler);
  std::signal(SIGTERM, stop_handler);
  ::unlink(socket_path.c_str());

  int server_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (server_fd < 0) {
    std::cerr << "Could not create worker socket\n";
    return 3;
  }

  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, socket_path.c_str(), sizeof(address.sun_path) - 1);
  if (::bind(server_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
    ::listen(server_fd, 1) != 0)
  {
    std::cerr << "Could not bind/listen on " << socket_path << ": " << std::strerror(errno) << '\n';
    ::close(server_fd);
    return 3;
  }

  try {
    cosmos_ros2_video_reasoner::TensorRTEdgeLLMConfig config;
    config.llm_engine_dir = argv[1];
    config.multimodal_engine_dir = argv[2];
    config.edge_llm_plugin_path = argv[3];
    config.jpeg_quality = jpeg_quality;
    cosmos_ros2_video_reasoner::TensorRTEdgeLLMBackend backend(config);
    backend.initialize();
    std::cout << "Cosmos inference worker ready on " << socket_path << std::endl;

    int client_fd = ::accept4(server_fd, nullptr, nullptr, SOCK_CLOEXEC);
    if (client_fd < 0) {throw std::runtime_error("worker accept failed");}

    while (!g_stop) {
      cosmos_ros2_video_reasoner::ipc::RequestHeader header;
      try {
        cosmos_ros2_video_reasoner::ipc::read_all(client_fd, &header, sizeof(header));
      } catch (std::exception const &) {
        break;
      }
      if (header.magic != cosmos_ros2_video_reasoner::ipc::kMagic ||
        header.version != cosmos_ros2_video_reasoner::ipc::kVersion ||
        header.encoding != cosmos_ros2_video_reasoner::ipc::kEncodingBgr8 ||
        header.width == 0 || header.height == 0 ||
        header.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        header.height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        static_cast<uint64_t>(header.step) !=
        static_cast<uint64_t>(header.width) * 3U ||
        header.prompt_bytes > cosmos_ros2_video_reasoner::ipc::kMaxTextBytes)
      {
        throw std::runtime_error("invalid IPC request header");
      }
      const uint64_t expected_image_bytes =
        static_cast<uint64_t>(header.step) * header.height;
      if (expected_image_bytes != header.image_bytes ||
        header.image_bytes > cosmos_ros2_video_reasoner::ipc::kMaxImageBytes)
      {
        throw std::runtime_error("invalid IPC image payload size");
      }

      std::vector<uint8_t> image_bytes(header.image_bytes);
      std::string prompt(header.prompt_bytes, '\0');
      cosmos_ros2_video_reasoner::ipc::read_all(client_fd, image_bytes.data(), image_bytes.size());
      cosmos_ros2_video_reasoner::ipc::read_all(client_fd, prompt.data(), prompt.size());

      cv::Mat view(
        static_cast<int>(header.height), static_cast<int>(header.width),
        CV_8UC3, image_bytes.data(), header.step);

      cosmos_ros2_video_reasoner::InferenceRequest request;
      request.image = view;
      request.prompt = std::move(prompt);
      request.max_generate_length = header.max_generate_length;
      request.temperature = header.temperature;
      request.top_p = header.top_p;
      request.top_k = header.top_k;

      cosmos_ros2_video_reasoner::InferenceResponse result;
      try {
        result = backend.infer(request);
      } catch (std::exception const & error) {
        result.success = false;
        result.error = error.what();
      }

      if (result.text.size() > cosmos_ros2_video_reasoner::ipc::kMaxTextBytes ||
        result.error.size() > cosmos_ros2_video_reasoner::ipc::kMaxTextBytes)
      {
        throw std::runtime_error("inference response exceeds IPC protocol limits");
      }

      cosmos_ros2_video_reasoner::ipc::ResponseHeader response;
      response.request_id = header.request_id;
      response.success = result.success ? 1U : 0U;
      response.text_bytes = static_cast<uint32_t>(result.text.size());
      response.error_bytes = static_cast<uint32_t>(result.error.size());
      response.inference_seconds = result.inference_seconds;
      cosmos_ros2_video_reasoner::ipc::write_all(client_fd, &response, sizeof(response));
      cosmos_ros2_video_reasoner::ipc::write_all(client_fd, result.text.data(), result.text.size());
      cosmos_ros2_video_reasoner::ipc::write_all(client_fd, result.error.data(), result.error.size());
    }
    ::close(client_fd);
  } catch (std::exception const & error) {
    std::cerr << "Inference worker failed: " << error.what() << '\n';
    ::close(server_fd);
    ::unlink(socket_path.c_str());
    return 4;
  }

  ::close(server_fd);
  ::unlink(socket_path.c_str());
  return 0;
}
