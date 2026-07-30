// Copyright 2025 cosmos_ros2_video_reasoner contributors
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

namespace cosmos_ros2_video_reasoner
{

/// Parameters for a single inference call.
struct InferenceRequest
{
  cv::Mat image;            //!< BGR image (OpenCV convention; backend converts as needed)
  std::string prompt;       //!< Text prompt forwarded to the VLM
  int max_generate_length;  //!< Maximum number of tokens to generate
  float temperature;        //!< Sampling temperature
  float top_p;              //!< Nucleus sampling probability
  int top_k;                //!< Top-k sampling parameter
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

}  // namespace cosmos_ros2_video_reasoner
