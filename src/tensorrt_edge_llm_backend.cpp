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

// ─── TensorRT Edge-LLM headers ───────────────────────────────────────────────
// These are resolved by the build system via TENSORRT_EDGE_LLM_ROOT.
#include "common/trtUtils.h"              // loadEdgellmPluginLib, DlDeleter
#include "runtime/llmInferenceRuntime.h"  // LLMInferenceRuntime, LLMGenerationRequest/Response
#include "runtime/imageUtils.h"           // imageUtils::loadImageFromMemory

#include "edge_vlm_ros/tensorrt_edge_llm_backend.hpp"

#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>

namespace edge_vlm_ros
{

TensorRTEdgeLLMBackend::TensorRTEdgeLLMBackend(TensorRTEdgeLLMConfig config)
: config_(std::move(config)),
  jpeg_quality_(config_.jpeg_quality)
{
}

TensorRTEdgeLLMBackend::~TensorRTEdgeLLMBackend()
{
  // Destroy the runtime before releasing the plugin library so that TensorRT
  // can call plugin destructors while the DSO is still loaded.
  runtime_.reset();

  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }

  if (plugin_handle_) {
    dlclose(plugin_handle_);
    plugin_handle_ = nullptr;
  }
}

void TensorRTEdgeLLMBackend::initialize()
{
  // ── 1. Validate engine paths ──────────────────────────────────────────
  if (config_.llm_engine_dir.empty()) {
    throw std::runtime_error("llm_engine_dir is required but was not provided");
  }
  if (config_.multimodal_engine_dir.empty()) {
    throw std::runtime_error("multimodal_engine_dir is required but was not provided");
  }

  if (jpeg_quality_ < 1 || jpeg_quality_ > 100) {
    throw std::runtime_error("jpeg_quality must be in [1, 100]");
  }

  namespace fs = std::filesystem;
  if (!fs::is_directory(config_.llm_engine_dir)) {
    throw std::runtime_error(
      "llm_engine_dir does not exist or is not a directory: " + config_.llm_engine_dir);
  }
  if (!fs::is_directory(config_.multimodal_engine_dir)) {
    throw std::runtime_error(
      "multimodal_engine_dir does not exist or is not a directory: " +
      config_.multimodal_engine_dir);
  }

  // ── 2. Load plugin library ────────────────────────────────────────────
  // Set EDGELLM_PLUGIN_PATH so that loadEdgellmPluginLib() picks it up.
  if (!config_.edge_llm_plugin_path.empty()) {
    if (!fs::exists(config_.edge_llm_plugin_path)) {
      throw std::runtime_error(
        "edge_llm_plugin_path does not exist: " + config_.edge_llm_plugin_path);
    }
    // Propagate to the environment so that trtUtils.h helper sees it.
    setenv("EDGELLM_PLUGIN_PATH", config_.edge_llm_plugin_path.c_str(), 1);
  }

  auto plugin_uptr = trt_edgellm::loadEdgellmPluginLib();
  if (!plugin_uptr) {
    throw std::runtime_error(
      "Failed to load TensorRT Edge-LLM plugin library. "
      "Check edge_llm_plugin_path and LD_LIBRARY_PATH.");
  }
  // Transfer ownership to a raw handle so we can manage the lifetime ourselves
  // (the runtime_ destructor must run first).
  plugin_handle_ = plugin_uptr.release();

  // ── 3. Create CUDA stream ─────────────────────────────────────────────
  // Match NVIDIA's llm_inference executable.  Edge-LLM's TensorRT runners use
  // auxiliary streams; a legacy blocking stream introduces implicit
  // default-stream synchronization and can stall multimodal preprocessing.
  if (cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreateWithFlags failed");
  }

  // ── 4. Construct the persistent LLMInferenceRuntime ───────────────────
  std::unordered_map<std::string, std::string> lora_weights_map;  // no LoRA for Cosmos
  try {
    runtime_ = std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
      config_.llm_engine_dir,
      config_.multimodal_engine_dir,
      lora_weights_map,
      cuda_stream_);
  } catch (const std::exception & e) {
    throw std::runtime_error(
      std::string("LLMInferenceRuntime construction failed: ") + e.what());
  }

  // ── 5. Capture CUDA decode graph (best-effort optimisation) ───────────
  if (!runtime_->captureDecodingCUDAGraph(cuda_stream_)) {
    // Non-fatal: fall back to plain execution context launch.
  }
}

InferenceResponse TensorRTEdgeLLMBackend::infer(const InferenceRequest & request)
{
  InferenceResponse resp;
  auto t0 = std::chrono::steady_clock::now();

  // ── Encode OpenCV Mat → JPEG bytes ────────────────────────────────────
  // TensorRT Edge-LLM ≤ 0.5.0 accepts image data through
  // rt::imageUtils::loadImageFromMemory(), which decodes stbi-compatible
  // formats (JPEG, PNG, BMP …).  We encode with JPEG for speed; quality is
  // controlled by the jpeg_quality_ field.
  std::vector<uchar> jpeg_buf;
  const std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
  if (!cv::imencode(".jpg", request.image, jpeg_buf, encode_params)) {
    resp.success = false;
    resp.error = "cv::imencode failed; cannot encode frame to JPEG";
    return resp;
  }

  // ── Load image into Edge-LLM ImageData ───────────────────────────────
  auto image_data = trt_edgellm::rt::imageUtils::loadImageFromMemory(
    jpeg_buf.data(), jpeg_buf.size());
  if (!image_data.buffer) {
    resp.success = false;
    resp.error = "loadImageFromMemory returned null buffer";
    return resp;
  }

  // ── Build multimodal generation request ──────────────────────────────
  trt_edgellm::rt::LLMGenerationRequest::Request inner_req;

  // ── System message (structured delivery only) ─────────────────────────
  // When a system message is present it is mapped to a native system Message.
  // Prior model outputs must not appear here — only operator-authored
  // instructions are eligible.
  // System prompts must be text-only; multimodal system messages are not
  // supported by the system-prompt cache and are rejected by the runtime.
  if (!request.system_message.empty()) {
    trt_edgellm::rt::Message sys_msg;
    sys_msg.role = "system";
    trt_edgellm::rt::Message::MessageContent sys_content;
    sys_content.type = "text";
    sys_content.content = request.system_message;
    sys_msg.contents.push_back(sys_content);
    inner_req.messages.push_back(std::move(sys_msg));
  }

  // ── Prior conversation turns (structured history) ─────────────────────
  // Each HistoryEntry represents one (user, assistant) exchange from prior
  // frames.  Historical user turns carry only text — no image — because we
  // do not retain image buffers across frames.  Assistant turns are untrusted
  // observations and must not influence system-level authority.
  for (const auto & entry : request.history) {
    trt_edgellm::rt::Message hist_user_msg;
    hist_user_msg.role = "user";
    trt_edgellm::rt::Message::MessageContent hist_user_content;
    hist_user_content.type = "text";
    hist_user_content.content = entry.user_text;
    hist_user_msg.contents.push_back(hist_user_content);
    inner_req.messages.push_back(std::move(hist_user_msg));

    trt_edgellm::rt::Message hist_asst_msg;
    hist_asst_msg.role = "assistant";
    trt_edgellm::rt::Message::MessageContent hist_asst_content;
    hist_asst_content.type = "text";
    hist_asst_content.content = entry.asst_text;
    hist_asst_msg.contents.push_back(hist_asst_content);
    inner_req.messages.push_back(std::move(hist_asst_msg));
  }

  // ── Current user message (image + task text) ──────────────────────────
  trt_edgellm::rt::Message user_msg;
  user_msg.role = "user";

  trt_edgellm::rt::Message::MessageContent image_content;
  image_content.type = "image";
  image_content.content = "<image>";  // placeholder; actual data in imageBuffers
  user_msg.contents.push_back(image_content);

  trt_edgellm::rt::Message::MessageContent text_content;
  text_content.type = "text";
  text_content.content = request.prompt;
  user_msg.contents.push_back(text_content);

  inner_req.messages.push_back(std::move(user_msg));
  inner_req.imageBuffers.push_back(std::move(image_data));

  trt_edgellm::rt::LLMGenerationRequest gen_req;
  gen_req.requests.push_back(std::move(inner_req));
  gen_req.temperature = request.temperature;
  gen_req.topP = request.top_p;
  gen_req.topK = static_cast<int64_t>(request.top_k);
  gen_req.maxGenerateLength = static_cast<int64_t>(request.max_generate_length);
  gen_req.applyChatTemplate = true;
  gen_req.addGenerationPrompt = true;

  // ── System-prompt cache (opt-in, hardware-validated) ──────────────────
  // The system-prompt cache accelerates TTFT for repeated requests that share
  // an identical, stable system prompt.  Cache keys are derived from the
  // system prompt text (and LoRA identity when applicable) and live in device
  // memory.  Multimodal system prompts are not cached.
  //
  // This flag is only effective when:
  //   1. request.system_message is non-empty,
  //   2. the pinned Edge-LLM version exposes the cache API, and
  //   3. the model engine was built with KV-cache reuse support.
  //
  // When the runtime does not support caching (older SDK or unsupported model),
  // the flag should be silently ignored — do not treat absence as a fatal error.
  //
  // Thor hardware validation is required to confirm the exact field name and
  // semantics exposed by the pinned TensorRT Edge-LLM version.
  // Uncomment and adapt the following block once the API is confirmed:
  //
  //   if (request.use_system_prompt_cache && !request.system_message.empty()) {
  //     gen_req.useSystemPromptCache = true;  // field name TBD; verify on Thor
  //   }
  //
  // Until validated, system-prompt caching is accepted by the configuration but
  // is not activated here.  The flag is propagated through IPC and is observable
  // in the prompt_config_hash for reproducibility.

  // ── Run inference ─────────────────────────────────────────────────────
  trt_edgellm::rt::LLMGenerationResponse gen_resp;
  const bool ok = runtime_->handleRequest(gen_req, gen_resp, cuda_stream_);

  auto t1 = std::chrono::steady_clock::now();
  resp.inference_seconds = std::chrono::duration<double>(t1 - t0).count();

  if (!ok || gen_resp.outputTexts.empty()) {
    resp.success = false;
    resp.error = "LLMInferenceRuntime::handleRequest returned false or empty output";
    return resp;
  }

  resp.success = true;
  resp.text = gen_resp.outputTexts[0];
  return resp;
}

}  // namespace edge_vlm_ros
