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

#include <memory>
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
class TensorRTEdgeLLMBackend : public InferenceBackend
{
public:
  explicit TensorRTEdgeLLMBackend(TensorRTEdgeLLMConfig config);
  ~TensorRTEdgeLLMBackend() override;

  /// Load the plugin library and construct the LLMInferenceRuntime.
  /// Throws std::runtime_error if any step fails.
  void initialize() override;

  /// Run VLM inference on the supplied image + prompt.
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

}  // namespace edge_vlm_ros
