// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

// ─── TensorRT Edge-LLM headers ───────────────────────────────────────────────
// These are resolved by the build system via TENSORRT_EDGE_LLM_ROOT.
#include "common/trtUtils.h"              // loadEdgellmPluginLib, DlDeleter
#include "runtime/llmInferenceRuntime.h"  // LLMInferenceRuntime, LLMGenerationRequest/Response
#include "runtime/imageUtils.h"           // imageUtils::loadImageFromMemory

#include "edge_vlm_ros/tensorrt_edge_llm_backend.hpp"

#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>

namespace edge_vlm_ros
{
namespace
{

double effective_video_fps_or_throw(const InferenceRequest & request)
{
  const auto fps = detail::infer_effective_video_fps(request);
  if (!fps.has_value() || !std::isfinite(*fps) || *fps <= 0.0) {
    throw std::runtime_error("unable to derive effective video fps for temporal/video request");
  }
  return *fps;
}

trt_edgellm::rt::imageUtils::ImageData build_native_video_imagedata(
  const std::vector<trt_edgellm::rt::imageUtils::ImageData> & frames, double fps,
  const std::vector<double> & timestamps)
{
  if (frames.empty()) {
    throw std::runtime_error("native video construction requires at least one frame");
  }

  const int64_t height = frames[0].height;
  const int64_t width = frames[0].width;
  const int64_t channels = frames[0].channels;
  if (height <= 0 || width <= 0 || channels != 3) {
    throw std::runtime_error("decoded primary frame has invalid dimensions for native video");
  }

  const int64_t frame_bytes = height * width * channels;
  const std::size_t frame_copy_bytes = static_cast<std::size_t>(frame_bytes);
  const int64_t total_frames = static_cast<int64_t>(frames.size());
  trt_edgellm::rt::Tensor stacked(
    {total_frames, height, width, channels},
    trt_edgellm::rt::DeviceType::kCPU,
    nvinfer1::DataType::kUINT8,
    "edge_vlm_ros::TensorRTEdgeLLMBackend::native_video_stacked");
  auto * dst = stacked.dataPointer<unsigned char>();

  for (std::size_t i = 0; i < frames.size(); ++i) {
    const auto & frame = frames[i];
    if (!frame.buffer) {
      throw std::runtime_error("decoded frame " + std::to_string(i) + " has null buffer");
    }
    if (frame.height != height || frame.width != width || frame.channels != channels) {
      throw std::runtime_error(
              "native video requires all frames to share identical dimensions/channels");
    }
    if (frame.bytesPerFrame() != frame_bytes) {
      throw std::runtime_error("decoded frame has unexpected byte size for native video copy");
    }
    std::memcpy(
      dst + static_cast<std::size_t>(i) * frame_copy_bytes,
      frame.data(),
      frame_copy_bytes);
  }

  trt_edgellm::rt::imageUtils::ImageData video(std::move(stacked));
  video.fps = fps;
  video.isVideo = true;
  if (!timestamps.empty()) {
    video.timestamps = timestamps;
  }
  return video;
}

}  // namespace


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
  resp.requested_sequence_type = temporal_sequence_type_to_string(request.sequence_type);
  try {
    detail::validate_temporal_metadata(request);
  } catch (const std::exception & e) {
    resp.success = false;
    resp.error = e.what();
    return resp;
  }
  const bool use_native_video = detail::uses_native_video_encoding(request);
  auto t0 = std::chrono::steady_clock::now();

  // ── Encode OpenCV Mat → JPEG bytes ────────────────────────────────────
  // TensorRT Edge-LLM ≤ 0.5.0 accepts image data through
  // rt::imageUtils::loadImageFromMemory(), which decodes stbi-compatible
  // formats (JPEG, PNG, BMP …).  We encode with JPEG for speed; quality is
  // controlled by the jpeg_quality_ field.
  //
  // Build the full ordered frame list: primary image first, then extra_images.
  // This is used for both content items and imageBuffers below.
  const std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};

  // Helper lambda: encode one Mat and load into an ImageData, or return error.
  auto encode_frame =
    [&](const cv::Mat & frame, std::size_t idx)
    -> std::pair<bool, trt_edgellm::rt::imageUtils::ImageData>
  {
    std::vector<uchar> jpeg_buf;
    if (!cv::imencode(".jpg", frame, jpeg_buf, encode_params)) {
      resp.success = false;
      resp.error = "cv::imencode failed for frame index " + std::to_string(idx);
      return {false, {}};
    }
    auto data = trt_edgellm::rt::imageUtils::loadImageFromMemory(
      jpeg_buf.data(), jpeg_buf.size());
    if (!data.buffer) {
      resp.success = false;
      resp.error = "loadImageFromMemory returned null buffer for frame index " +
                   std::to_string(idx);
      return {false, {}};
    }
    return {true, std::move(data)};
  };

  // Encode the primary image (index 0).
  auto [ok0, primary_data] = encode_frame(request.image, 0);
  if (!ok0) {
    return resp;
  }

  // Encode extra images (indices 1 … N) and fail clearly on any invalid input.
  std::vector<trt_edgellm::rt::imageUtils::ImageData> decoded_frames;
  if (use_native_video) {
    decoded_frames.reserve(1U + request.extra_images.size());
    decoded_frames.push_back(std::move(primary_data));
  }
  std::vector<trt_edgellm::rt::imageUtils::ImageData> extra_data;
  if (!use_native_video) {
    extra_data.reserve(request.extra_images.size());
  }
  for (std::size_t i = 0; i < request.extra_images.size(); ++i) {
    if (request.extra_images[i].empty()) {
      resp.success = false;
      resp.error = "extra_images[" + std::to_string(i) + "] is an empty Mat";
      return resp;
    }
    auto [ok_i, data_i] = encode_frame(request.extra_images[i], i + 1);
    if (!ok_i) {
      return resp;
    }
    if (use_native_video) {
      decoded_frames.push_back(std::move(data_i));
    } else {
      extra_data.push_back(std::move(data_i));
    }
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

  // ── Current user message (media item(s) + task text) ───────────────────
  trt_edgellm::rt::Message user_msg;
  user_msg.role = "user";

  auto push_media_content = [&](const char * type, const char * placeholder) {
    trt_edgellm::rt::Message::MessageContent c;
    c.type = type;
    c.content = placeholder;
    user_msg.contents.push_back(c);
  };

  if (use_native_video) {
    push_media_content("video", "<video>");
  } else {
    push_media_content("image", "<image>");
    for (std::size_t i = 0; i < request.extra_images.size(); ++i) {
      push_media_content("image", "<image>");
    }
  }

  trt_edgellm::rt::Message::MessageContent text_content;
  text_content.type = "text";
  text_content.content = request.prompt;
  user_msg.contents.push_back(text_content);

  inner_req.messages.push_back(std::move(user_msg));

  if (use_native_video) {
    try {
      auto video = build_native_video_imagedata(
        decoded_frames,
        effective_video_fps_or_throw(request),
        request.frame_timestamps_sec);
      inner_req.imageBuffers.push_back(std::move(video));
    } catch (const std::exception & e) {
      resp.success = false;
      resp.error = std::string("failed to represent temporal/video request natively: ") + e.what();
      return resp;
    }
    resp.runtime_temporal_encoding = "native_qwen3vl_video_imagedata_mrope_timestamps";
    resp.temporal_fallback_used = false;
  } else {
    // imageBuffers: primary first, then extra frames in temporal order.
    inner_req.imageBuffers.push_back(std::move(primary_data));
    inner_req.imageBuffers.insert(
      inner_req.imageBuffers.end(),
      std::make_move_iterator(extra_data.begin()),
      std::make_move_iterator(extra_data.end()));
    resp.runtime_temporal_encoding = "ordered_multi_image_no_native_temporal_metadata";
    resp.temporal_fallback_used = false;
  }

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
