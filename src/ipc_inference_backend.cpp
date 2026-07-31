// Copyright 2025 cosmos_ros2_video_reasoner contributors
#include "cosmos_ros2_video_reasoner/ipc_inference_backend.hpp"
#include "cosmos_ros2_video_reasoner/ipc_protocol.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace cosmos_ros2_video_reasoner
{
IpcInferenceBackend::IpcInferenceBackend(IpcInferenceConfig config)
: config_(std::move(config))
{
}

IpcInferenceBackend::~IpcInferenceBackend()
{
  close_connection();
}

void IpcInferenceBackend::close_connection() noexcept
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

void IpcInferenceBackend::initialize()
{
  connect_worker();
}

void IpcInferenceBackend::connect_worker()
{
  close_connection();
  if (config_.connect_timeout_seconds <= 0 || config_.request_timeout_seconds <= 0) {
    throw std::runtime_error("worker timeouts must be positive");
  }
  if (config_.socket_path.empty() ||
    config_.socket_path.size() >= sizeof(sockaddr_un::sun_path))
  {
    throw std::runtime_error("invalid worker socket path");
  }

  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::seconds(config_.connect_timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
      throw std::runtime_error("failed to create IPC socket");
    }

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strncpy(address.sun_path, config_.socket_path.c_str(), sizeof(address.sun_path) - 1);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) {
      timeval timeout{};
      timeout.tv_sec = config_.request_timeout_seconds;
      if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0)
      {
        const int error = errno;
        ::close(fd);
        throw std::runtime_error(
                "failed to configure IPC request timeout: " +
                std::string(std::strerror(error)));
      }
      socket_fd_ = fd;
      return;
    }

    ::close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  throw std::runtime_error("timed out connecting to inference worker: " + config_.socket_path);
}

InferenceResponse IpcInferenceBackend::infer(InferenceRequest const & request)
{
  if (socket_fd_ < 0) {
    connect_worker();
  }
  if (request.image.empty() || request.image.type() != CV_8UC3) {
    throw std::runtime_error("IPC backend requires a non-empty CV_8UC3 BGR image");
  }

  cv::Mat packed = request.image.isContinuous() ? request.image : request.image.clone();
  const size_t image_size = packed.total() * packed.elemSize();
  const size_t max_image_bytes = std::min(config_.max_image_bytes, static_cast<size_t>(ipc::kMaxImageBytes));
  const size_t max_text_bytes = std::min(config_.max_text_bytes, static_cast<size_t>(ipc::kMaxTextBytes));
  if (image_size > max_image_bytes || request.prompt.size() > max_text_bytes) {
    throw std::runtime_error("IPC request exceeds protocol limits");
  }

  ipc::RequestHeader header;
  header.request_id = next_request_id_++;
  header.width = static_cast<uint32_t>(packed.cols);
  header.height = static_cast<uint32_t>(packed.rows);
  header.step = static_cast<uint32_t>(packed.cols * 3);
  header.image_bytes = static_cast<uint32_t>(image_size);
  header.prompt_bytes = static_cast<uint32_t>(request.prompt.size());
  header.max_generate_length = request.max_generate_length;
  header.temperature = request.temperature;
  header.top_p = request.top_p;
  header.top_k = request.top_k;

  ipc::ResponseHeader response_header;
  try {
    ipc::write_all(socket_fd_, &header, sizeof(header));
    ipc::write_all(socket_fd_, packed.data, image_size);
    ipc::write_all(socket_fd_, request.prompt.data(), request.prompt.size());
    ipc::read_all(socket_fd_, &response_header, sizeof(response_header));
  } catch (...) {
    close_connection();
    throw;
  }
  if (response_header.magic != ipc::kMagic || response_header.version != ipc::kVersion ||
    response_header.request_id != header.request_id ||
    response_header.text_bytes > ipc::kMaxTextBytes ||
    response_header.error_bytes > ipc::kMaxTextBytes)
  {
    close_connection();
    throw std::runtime_error("invalid inference worker response");
  }

  InferenceResponse response;
  response.success = response_header.success != 0;
  response.inference_seconds = response_header.inference_seconds;
  response.text.resize(response_header.text_bytes);
  response.error.resize(response_header.error_bytes);
  try {
    ipc::read_all(socket_fd_, response.text.data(), response.text.size());
    ipc::read_all(socket_fd_, response.error.data(), response.error.size());
  } catch (...) {
    close_connection();
    throw;
  }
  return response;
}
}  // namespace cosmos_ros2_video_reasoner
