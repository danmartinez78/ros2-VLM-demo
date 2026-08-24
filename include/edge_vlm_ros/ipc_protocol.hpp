// Copyright 2025 edge_vlm_ros contributors
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <sys/socket.h>
#include <unistd.h>

namespace edge_vlm_ros::ipc
{
constexpr uint32_t kMagic = 0x45564C4D;  // EVLM
/// IPC schema version.
/// v1: prompt_bytes only, single inline user message.
/// v2: schema_flags, system_bytes, history_count — adds structured message roles.
/// v3: sequence_type/fps/frame_timestamps metadata + response temporal encoding fields.
constexpr uint32_t kVersion = 3;
constexpr uint32_t kEncodingBgr8 = 1;
constexpr uint32_t kMaxTextBytes = 1024 * 1024;
constexpr uint32_t kMaxImageBytes = 256 * 1024 * 1024;
constexpr uint32_t kMaxHistoryEntries = 256;

/// schema_flags bit definitions.
/// kSchemaFlagInline (0): legacy inline delivery — payload is [image][prompt].
/// kSchemaFlagStructured: structured delivery — payload is
///   [image][system_bytes bytes][prompt_bytes bytes][history_count × entry].
/// kSchemaFlagSysCache: worker should attempt system-prompt caching for this
///   request (only valid when kSchemaFlagStructured is set and system_bytes > 0).
///   Cache eligibility and availability are validated on the worker side; the
///   flag is silently ignored when the runtime does not support caching.
constexpr uint32_t kSchemaFlagInline = 0U;
constexpr uint32_t kSchemaFlagStructured = 1U << 0;
constexpr uint32_t kSchemaFlagSysCache = 1U << 1;
/// kSchemaFlagMultiImage: request carries multiple images in temporal order.
/// When set, RequestHeader.image_count holds the total image count (>= 2).
/// Wire format after RequestHeader:
///   [image_count-1 × PerImageHeader]          — headers for extra images (index 1..N-1)
///   [image_0_bytes]                            — raw BGR data for image 0 (primary)
///   [image_1_bytes .. image_{N-1}_bytes]       — raw BGR data for extra images
///   [timestamp_count × double]                 — optional frame timestamps (when flagged)
///   [system_bytes][prompt_bytes][history ...]  — normal structured/inline payload
constexpr uint32_t kSchemaFlagMultiImage = 1U << 2;
/// kSchemaFlagHasFps: RequestHeader.fps is populated with a finite value > 0.
constexpr uint32_t kSchemaFlagHasFps = 1U << 3;
/// kSchemaFlagHasFrameTimestamps: request carries `timestamp_count` doubles after
/// image payload (and before prompt/system/history payload).
constexpr uint32_t kSchemaFlagHasFrameTimestamps = 1U << 4;
constexpr uint32_t kMaxExtraImages = 31U;  ///< Maximum extra images beyond the primary.

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
  /// User-message bytes (or full inline prompt bytes when schema_flags == kSchemaFlagInline).
  uint32_t prompt_bytes{0};
  int32_t max_generate_length{0};
  float temperature{0.0F};
  float top_p{0.0F};
  int32_t top_k{0};
  /// Delivery-mode flags (see kSchemaFlag* constants). Default 0 = inline.
  uint32_t schema_flags{kSchemaFlagInline};
  /// System-message byte count (0 when schema_flags == kSchemaFlagInline).
  uint32_t system_bytes{0};
  /// Number of prior (user, assistant) history turns that follow the prompt payload.
  uint32_t history_count{0};
  /// Single-image requests: 0 (reserved).
  /// Multi-image requests (kSchemaFlagMultiImage set): total image count including primary (>= 2).
  uint32_t image_count{0};
  /// Requested sequence semantics: 0=images, 1=temporal_images, 2=video.
  uint32_t sequence_type{0U};
  /// Optional fps value. Valid when kSchemaFlagHasFps is set.
  double fps{0.0};
  /// Number of entries in frame_timestamps_sec sent on the wire after image payload.
  uint32_t timestamp_count{0U};
  /// Reserved for forward-compatible protocol extension.
  uint32_t reserved{0U};
};

/// Fixed-size header preceding each history entry in the structured wire format.
/// Layout per entry: [HistoryEntryHeader][user_bytes bytes][asst_bytes bytes]
struct HistoryEntryHeader
{
  uint32_t user_bytes{0};
  uint32_t asst_bytes{0};
};

/// Per-image header used only when kSchemaFlagMultiImage is set.
/// One PerImageHeader precedes each extra image (images 1..N-1).
/// The primary image (index 0) is described by RequestHeader.width/height/step/image_bytes.
struct PerImageHeader
{
  uint32_t width{0};
  uint32_t height{0};
  uint32_t step{0};
  uint32_t image_bytes{0};
};
static_assert(std::is_trivially_copyable_v<PerImageHeader>);

struct ResponseHeader
{
  uint32_t magic{kMagic};
  uint32_t version{kVersion};
  uint64_t request_id{0};
  uint32_t success{0};
  uint32_t text_bytes{0};
  uint32_t error_bytes{0};
  uint32_t temporal_encoding_bytes{0};
  uint32_t temporal_fallback_used{0};
  double inference_seconds{0.0};
};

static_assert(std::is_trivially_copyable_v<RequestHeader>);
static_assert(std::is_trivially_copyable_v<ResponseHeader>);
static_assert(std::is_trivially_copyable_v<HistoryEntryHeader>);

inline void write_all(int fd, void const * data, size_t size)
{
  auto const * bytes = static_cast<uint8_t const *>(data);
  while (size > 0) {
    const ssize_t written = ::send(fd, bytes, size, MSG_NOSIGNAL);
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
}  // namespace edge_vlm_ros::ipc
