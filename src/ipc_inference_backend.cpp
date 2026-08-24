// Copyright 2025 edge_vlm_ros contributors
#include "edge_vlm_ros/ipc_inference_backend.hpp"
#include "edge_vlm_ros/ipc_protocol.hpp"

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

namespace edge_vlm_ros
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
  if (request.extra_images.size() > ipc::kMaxExtraImages) {
    throw std::runtime_error("IPC request exceeds maximum extra image count");
  }
  detail::validate_temporal_metadata(request);

  const size_t max_image_bytes = std::min(
    config_.max_image_bytes, static_cast<size_t>(ipc::kMaxImageBytes));
  const size_t max_text_bytes = std::min(
    config_.max_text_bytes, static_cast<size_t>(ipc::kMaxTextBytes));

  // Pack primary image.
  cv::Mat packed = request.image.isContinuous() ? request.image : request.image.clone();
  const size_t image_size = packed.total() * packed.elemSize();

  // Pack extra images and validate.
  std::vector<cv::Mat> packed_extra;
  packed_extra.reserve(request.extra_images.size());
  for (size_t ei = 0; ei < request.extra_images.size(); ++ei) {
    const auto & img = request.extra_images[ei];
    if (img.empty() || img.type() != CV_8UC3) {
      throw std::runtime_error(
              "IPC backend: extra image " + std::to_string(ei) +
              " must be non-empty CV_8UC3 BGR");
    }
    packed_extra.push_back(img.isContinuous() ? img : img.clone());
    const size_t sz = packed_extra.back().total() * packed_extra.back().elemSize();
    if (sz > max_image_bytes) {
      throw std::runtime_error(
              "IPC request: extra image " + std::to_string(ei) +
              " exceeds protocol limits (" + std::to_string(sz) + " bytes)");
    }
  }

  if (image_size > max_image_bytes) {
    throw std::runtime_error("IPC request exceeds protocol limits");
  }
  if (request.prompt.size() > max_text_bytes) {
    throw std::runtime_error("IPC request exceeds protocol limits");
  }
  if (request.system_message.size() > max_text_bytes) {
    throw std::runtime_error("IPC request exceeds protocol limits");
  }
  if (request.history.size() > ipc::kMaxHistoryEntries) {
    throw std::runtime_error("IPC request history exceeds protocol limits");
  }
  for (const auto & entry : request.history) {
    if (entry.user_text.size() > max_text_bytes || entry.asst_text.size() > max_text_bytes) {
      throw std::runtime_error("IPC request history entry exceeds protocol limits");
    }
  }

  // Determine schema flags based on request content.
  // Structured mode is used whenever a system message or history is present.
  const bool has_structured = !request.system_message.empty() || !request.history.empty();
  const bool has_extra = !packed_extra.empty();
  uint32_t schema_flags = ipc::kSchemaFlagInline;
  if (has_structured) {
    schema_flags |= ipc::kSchemaFlagStructured;
    if (request.use_system_prompt_cache && !request.system_message.empty()) {
      schema_flags |= ipc::kSchemaFlagSysCache;
    }
  }
  if (has_extra) {
    schema_flags |= ipc::kSchemaFlagMultiImage;
  }
  if (request.fps.has_value()) {
    schema_flags |= ipc::kSchemaFlagHasFps;
  }
  if (!request.frame_timestamps_sec.empty()) {
    schema_flags |= ipc::kSchemaFlagHasFrameTimestamps;
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
  header.schema_flags = schema_flags;
  header.system_bytes = static_cast<uint32_t>(request.system_message.size());
  header.history_count = static_cast<uint32_t>(request.history.size());
  header.image_count = has_extra ? static_cast<uint32_t>(1 + packed_extra.size()) : 0U;
  header.sequence_type = static_cast<uint32_t>(request.sequence_type);
  header.fps = request.fps.value_or(0.0);
  header.timestamp_count = static_cast<uint32_t>(request.frame_timestamps_sec.size());

  ipc::ResponseHeader response_header;
  try {
    ipc::write_all(socket_fd_, &header, sizeof(header));
    // Multi-image: send PerImageHeaders for extra images, then all image data.
    if (has_extra) {
      for (const auto & ei : packed_extra) {
        ipc::PerImageHeader pih;
        pih.width = static_cast<uint32_t>(ei.cols);
        pih.height = static_cast<uint32_t>(ei.rows);
        pih.step = static_cast<uint32_t>(ei.cols * 3);
        pih.image_bytes = static_cast<uint32_t>(ei.total() * ei.elemSize());
        ipc::write_all(socket_fd_, &pih, sizeof(pih));
      }
    }
    ipc::write_all(socket_fd_, packed.data, image_size);
    if (has_extra) {
      for (const auto & ei : packed_extra) {
        ipc::write_all(socket_fd_, ei.data, ei.total() * ei.elemSize());
      }
    }
    if (!request.frame_timestamps_sec.empty()) {
      ipc::write_all(
        socket_fd_,
        request.frame_timestamps_sec.data(),
        request.frame_timestamps_sec.size() * sizeof(double));
    }
    if (has_structured && !request.system_message.empty()) {
      ipc::write_all(socket_fd_, request.system_message.data(), request.system_message.size());
    }
    ipc::write_all(socket_fd_, request.prompt.data(), request.prompt.size());
    if (has_structured) {
      for (const auto & entry : request.history) {
        ipc::HistoryEntryHeader entry_header;
        entry_header.user_bytes = static_cast<uint32_t>(entry.user_text.size());
        entry_header.asst_bytes = static_cast<uint32_t>(entry.asst_text.size());
        ipc::write_all(socket_fd_, &entry_header, sizeof(entry_header));
        ipc::write_all(socket_fd_, entry.user_text.data(), entry.user_text.size());
        ipc::write_all(socket_fd_, entry.asst_text.data(), entry.asst_text.size());
      }
    }
    ipc::read_all(socket_fd_, &response_header, sizeof(response_header));
  } catch (...) {
    close_connection();
    throw;
  }
  if (response_header.magic != ipc::kMagic || response_header.version != ipc::kVersion ||
    response_header.request_id != header.request_id ||
    response_header.text_bytes > ipc::kMaxTextBytes ||
    response_header.error_bytes > ipc::kMaxTextBytes ||
    response_header.temporal_encoding_bytes > ipc::kMaxTextBytes)
  {
    close_connection();
    throw std::runtime_error("invalid inference worker response");
  }

  InferenceResponse response;
  response.success = response_header.success != 0;
  response.inference_seconds = response_header.inference_seconds;
  response.text.resize(response_header.text_bytes);
  response.error.resize(response_header.error_bytes);
  response.runtime_temporal_encoding.resize(response_header.temporal_encoding_bytes);
  response.temporal_fallback_used = response_header.temporal_fallback_used != 0U;
  response.requested_sequence_type = temporal_sequence_type_to_string(request.sequence_type);
  try {
    ipc::read_all(socket_fd_, response.text.data(), response.text.size());
    ipc::read_all(socket_fd_, response.error.data(), response.error.size());
    ipc::read_all(
      socket_fd_, response.runtime_temporal_encoding.data(),
      response.runtime_temporal_encoding.size());
  } catch (...) {
    close_connection();
    throw;
  }
  return response;
}
}  // namespace edge_vlm_ros
