// Copyright 2025 cosmos_ros2_video_reasoner contributors
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>

namespace cosmos_ros2_video_reasoner::ipc
{
constexpr uint32_t kMagic = 0x434F534D;  // COSM
constexpr uint32_t kVersion = 1;
constexpr uint32_t kEncodingBgr8 = 1;
constexpr uint32_t kMaxTextBytes = 1024 * 1024;

struct RequestHeader
{
  uint32_t magic{kMagic};
  uint32_t version{kVersion};
  uint64_t request_id{0};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t step{0};
  uint32_t encoding{kEncodingBgr8};
  uint32_t image_bytes{0};
  uint32_t prompt_bytes{0};
  int32_t max_generate_length{0};
  float temperature{0.0F};
  float top_p{0.0F};
  int32_t top_k{0};
};

struct ResponseHeader
{
  uint32_t magic{kMagic};
  uint32_t version{kVersion};
  uint64_t request_id{0};
  uint32_t success{0};
  uint32_t text_bytes{0};
  uint32_t error_bytes{0};
  double inference_seconds{0.0};
};

static_assert(std::is_trivially_copyable_v<RequestHeader>);
static_assert(std::is_trivially_copyable_v<ResponseHeader>);

inline void write_all(int fd, void const * data, size_t size)
{
  auto const * bytes = static_cast<uint8_t const *>(data);
  while (size > 0) {
    const ssize_t written = ::write(fd, bytes, size);
    if (written < 0) {
      if (errno == EINTR) {continue;}
      throw std::runtime_error("IPC write failed: " + std::string(std::strerror(errno)));
    }
    if (written == 0) {throw std::runtime_error("IPC peer closed during write");}
    bytes += written;
    size -= static_cast<size_t>(written);
  }
}

inline void read_all(int fd, void * data, size_t size)
{
  auto * bytes = static_cast<uint8_t *>(data);
  while (size > 0) {
    const ssize_t received = ::read(fd, bytes, size);
    if (received < 0) {
      if (errno == EINTR) {continue;}
      throw std::runtime_error("IPC read failed: " + std::string(std::strerror(errno)));
    }
    if (received == 0) {throw std::runtime_error("IPC peer closed");}
    bytes += received;
    size -= static_cast<size_t>(received);
  }
}
}  // namespace cosmos_ros2_video_reasoner::ipc
