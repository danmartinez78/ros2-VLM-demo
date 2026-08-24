// Copyright 2025 edge_vlm_ros contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "edge_vlm_ros/inference_backend.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

// ─── Forward declarations for TensorRT Edge-LLM types ──────────────────────
// These headers ship with the TensorRT-Edge-LLM SDK and are resolved at
// build time via TENSORRT_EDGE_LLM_ROOT / TENSORRT_EDGE_LLM_BUILD_DIR.
namespace trt_edgellm
{
namespace rt
{
class LLMInferenceRuntime;
}  // namespace rt
}  // namespace trt_edgellm

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace edge_vlm_ros
{

/// Configuration for the TensorRT Edge-LLM backend.
struct TensorRTEdgeLLMConfig
{
  std::string llm_engine_dir;          //!< Path to the language-model TensorRT engine directory
  std::string multimodal_engine_dir;   //!< Path to the visual encoder TensorRT engine directory
  std::string edge_llm_plugin_path;    //!< Absolute path to libNvInfer_edgellm_plugin.so
  int jpeg_quality{90};                 //!< JPEG quality for in-memory frame encoding
};

/// Production backend that wraps trt_edgellm::rt::LLMInferenceRuntime.
///
/// The runtime and CUDA stream are created once in initialize() and reused
/// for all infer() calls.  The plugin library is loaded via dlopen before
/// the runtime is constructed, as required by TensorRT Edge-LLM.
///
/// Image handling
/// ───────────────
/// TensorRT Edge-LLM ≤ 0.5.0 accepts image data through
/// rt::imageUtils::loadImageFromMemory(), which reads a stbi-compatible
/// byte buffer (JPEG/PNG).  This backend encodes the OpenCV Mat to JPEG
/// bytes with cv::imencode() and passes the result directly to
/// loadImageFromMemory(), avoiding any temporary file on disk.
///
/// Multi-frame support
/// ────────────────────
/// - sequence_type=images: one "image" content item + one imageBuffers entry
///   per frame (primary first, then extra_images in temporal order).
/// - sequence_type=temporal_images|video: one native "video" content item and
///   one stacked ImageData buffer with shape [T,H,W,3], isVideo=true, fps, and
///   optional source timestamps.
class TensorRTEdgeLLMBackend : public InferenceBackend
{
public:
  explicit TensorRTEdgeLLMBackend(TensorRTEdgeLLMConfig config);
  ~TensorRTEdgeLLMBackend() override;

  /// Load the plugin library and construct the LLMInferenceRuntime.
  /// Throws std::runtime_error if any step fails.
  void initialize() override;

  /// Run VLM inference on the supplied image(s) + prompt.
  InferenceResponse infer(const InferenceRequest & request) override;

private:
  TensorRTEdgeLLMConfig config_;

  // plugin library handle (must outlive the runtime)
  void * plugin_handle_{nullptr};

  // CUDA stream shared across all inference calls
  cudaStream_t cuda_stream_{nullptr};

  // Persistent inference runtime (non-null after initialize())
  std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> runtime_;

  // JPEG quality used when encoding frames for the VLM
  int jpeg_quality_{90};
};

namespace detail
{

inline bool uses_native_video_encoding(const InferenceRequest & request) noexcept
{
  return request.sequence_type != TemporalSequenceType::kImages;
}

inline std::optional<double> infer_effective_video_fps(const InferenceRequest & request) noexcept
{
  if (!uses_native_video_encoding(request)) {
    return std::nullopt;
  }
  if (request.fps.has_value()) {
    return request.fps;
  }
  if (request.frame_timestamps_sec.size() >= 2U) {
    const double span = request.frame_timestamps_sec.back() - request.frame_timestamps_sec.front();
    if (span > 0.0) {
      return static_cast<double>(frame_count(request) - 1U) / span;
    }
  }
  return 1.0;
}

/// Returns the expected number of media content items for a request.
/// `images` => one "image" item per frame.
/// `temporal_images`/`video` => one "video" item for the stacked frame tensor.
inline std::size_t media_content_count(const InferenceRequest & request) noexcept
{
  return uses_native_video_encoding(request) ? 1U : frame_count(request);
}

/// Returns the expected media content type for the current user message.
inline const char * media_content_type(const InferenceRequest & request) noexcept
{
  return uses_native_video_encoding(request) ? "video" : "image";
}

/// Returns the expected number of imageBuffers entries the backend will push.
inline std::size_t image_buffer_count(const InferenceRequest & request) noexcept
{
  return media_content_count(request);
}

/// Returns the expected total number of content items in the user message:
/// media items plus one text item.
inline std::size_t user_message_content_count(const InferenceRequest & request) noexcept
{
  return media_content_count(request) + 1U;  // +1 for the text item
}

}  // namespace detail

}  // namespace edge_vlm_ros
