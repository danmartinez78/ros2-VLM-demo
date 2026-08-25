// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace edge_vlm_ros
{

enum class TemporalSequenceType : uint32_t
{
  kImages = 0U,
  kTemporalImages = 1U,
  kVideo = 2U,
};

inline const char * temporal_sequence_type_to_string(TemporalSequenceType type) noexcept
{
  switch (type) {
    case TemporalSequenceType::kImages:
      return "images";
    case TemporalSequenceType::kTemporalImages:
      return "temporal_images";
    case TemporalSequenceType::kVideo:
      return "video";
    default:
      return "unknown";
  }
}

/// A single prior conversation turn carried through IPC into native message structures.
///
/// user_text: the effective user-message text rendered for that frame (may be the full
///   inline prompt when the entry was created in inline mode, or the task portion only
///   when created in structured mode — either is safe as a historical user turn).
/// asst_text: the model output for that frame.  This is an untrusted observation;
///   it must never be promoted into a system message.
struct HistoryEntry
{
  std::string user_text;
  std::string asst_text;
};

/// Parameters for a single inference call.
struct InferenceRequest
{
  cv::Mat image;            //!< BGR image (OpenCV convention; backend converts as needed)
  /// Additional images in temporal order (index 0 = first extra, after `image`).
  /// When non-empty, the IPC backend sends a kSchemaFlagMultiImage request carrying
  /// `image` + `extra_images` as a single multi-frame inference call.
  std::vector<cv::Mat> extra_images;
  /// Requested sequence contract.
  TemporalSequenceType sequence_type{TemporalSequenceType::kImages};
  /// Optional sequence rate in frames/second. Must be finite and > 0 when set.
  std::optional<double> fps;
  /// Optional frame timestamp vector in seconds (one per frame, strictly increasing).
  std::vector<double> frame_timestamps_sec;
  std::string prompt;       //!< User-message text (task prompt or full inline prompt)
  int max_generate_length;  //!< Maximum number of tokens to generate
  float temperature;        //!< Sampling temperature
  float top_p;              //!< Nucleus sampling probability
  int top_k;                //!< Top-k sampling parameter

  /// Optional system-role message (empty = no system message / inline delivery).
  /// When non-empty, the backend maps this to a native system Message.
  /// Must not contain prior model outputs; system instructions only.
  std::string system_message;

  /// Prior conversation turns, ordered oldest-first.
  /// Only populated when instruction_delivery_mode is "structured" and
  /// prompt_history_max_entries > 0.  Each entry represents one (user, assistant)
  /// exchange prior to the current frame.  The assistant text is an untrusted
  /// observation and must not be promoted to system-level authority.
  std::vector<HistoryEntry> history;

  /// When true, request the worker to attempt system-prompt caching for this
  /// request.  Valid only when system_message is non-empty and the runtime/model
  /// supports the feature.  Silently ignored when unavailable.
  bool use_system_prompt_cache{false};
};

/// Result returned from a single inference call.
struct InferenceResponse
{
  bool success{false};         //!< True when inference completed without error
  std::string text;            //!< Generated text (valid when success == true)
  std::string error;           //!< Error description (valid when success == false)
  double inference_seconds{0}; //!< Wall-clock time spent in inference
  std::string requested_sequence_type{"images"};  //!< Caller-requested sequence contract.
  std::string runtime_temporal_encoding{
    "ordered_multi_image_no_native_temporal_metadata"};  //!< Runtime representation used.
  bool temporal_fallback_used{false};  //!< True when temporal/video request degraded to ordered images.
};

/// Abstract interface for VLM inference backends.
///
/// Implementations must be thread-safe for the public methods below.
class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;

  /// Load engines and allocate device resources.
  /// Called once during node startup before any infer() calls.
  /// Throws std::runtime_error on failure.
  virtual void initialize() = 0;

  /// Run a single inference pass.
  /// May be called from any thread but is never called concurrently.
  virtual InferenceResponse infer(const InferenceRequest & request) = 0;
};

namespace detail
{

inline std::size_t frame_count(const InferenceRequest & request) noexcept
{
  return 1u + request.extra_images.size();
}

inline void validate_temporal_metadata(const InferenceRequest & request)
{
  if (request.fps.has_value()) {
    const double value = *request.fps;
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::runtime_error("fps must be a finite value > 0");
    }
  }

  const auto total_frames = frame_count(request);
  if (!request.frame_timestamps_sec.empty()) {
    if (request.frame_timestamps_sec.size() != total_frames) {
      throw std::runtime_error(
              "frame_timestamps_sec size must match total frame count (" +
              std::to_string(total_frames) + ")");
    }
    for (std::size_t i = 0; i < request.frame_timestamps_sec.size(); ++i) {
      const double ts = request.frame_timestamps_sec[i];
      if (!std::isfinite(ts)) {
        throw std::runtime_error("frame_timestamps_sec must contain only finite values");
      }
      if (i > 0 && !(ts > request.frame_timestamps_sec[i - 1])) {
        throw std::runtime_error("frame_timestamps_sec must be strictly increasing");
      }
    }
  }

  if (request.sequence_type == TemporalSequenceType::kImages) {
    if (request.fps.has_value() || !request.frame_timestamps_sec.empty()) {
      throw std::runtime_error(
              "sequence_type=images must not include fps or frame_timestamps_sec");
    }
    return;
  }

  if (request.fps.has_value() && request.frame_timestamps_sec.size() >= 2U) {
    const double expected_dt = 1.0 / *request.fps;
    constexpr double kEpsilon = 1e-3;
    for (std::size_t i = 1; i < request.frame_timestamps_sec.size(); ++i) {
      const double dt = request.frame_timestamps_sec[i] - request.frame_timestamps_sec[i - 1];
      if (std::fabs(dt - expected_dt) > kEpsilon) {
        throw std::runtime_error(
                "fps conflicts with frame_timestamps_sec spacing at index " + std::to_string(i));
      }
    }
  }
}

}  // namespace detail

}  // namespace edge_vlm_ros
