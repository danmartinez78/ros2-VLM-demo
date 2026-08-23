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

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace edge_vlm_ros
{

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

}  // namespace edge_vlm_ros
