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

#include "cosmos_ros2_video_reasoner/inference_backend.hpp"

#include <chrono>
#include <functional>
#include <string>
#include <thread>

namespace cosmos_ros2_video_reasoner
{

/// Fake backend used in unit tests.
///
/// By default it returns an immediate successful response.  Tests can
/// override behaviour by supplying a custom handler function.
class FakeInferenceBackend : public InferenceBackend
{
public:
  using Handler = std::function<InferenceResponse(const InferenceRequest &)>;

  /// Default: instant success with a fixed text reply.
  FakeInferenceBackend() = default;

  /// Custom handler; useful for simulating failures or slow inference.
  explicit FakeInferenceBackend(Handler handler)
  : handler_(std::move(handler)) {}

  void initialize() override
  {
    initialized_ = true;
  }

  InferenceResponse infer(const InferenceRequest & request) override
  {
    if (handler_) {
      return handler_(request);
    }
    InferenceResponse resp;
    resp.success = true;
    resp.text = "fake response";
    resp.inference_seconds = 0.001;
    return resp;
  }

  bool is_initialized() const noexcept {return initialized_;}

private:
  Handler handler_;
  bool initialized_{false};
};

/// Slow fake backend: sleeps for `delay_ms` milliseconds before returning.
class SlowFakeInferenceBackend : public InferenceBackend
{
public:
  explicit SlowFakeInferenceBackend(std::chrono::milliseconds delay)
  : delay_(delay) {}

  void initialize() override {}

  InferenceResponse infer(const InferenceRequest & /*request*/) override
  {
    std::this_thread::sleep_for(delay_);
    InferenceResponse resp;
    resp.success = true;
    resp.text = "slow response";
    resp.inference_seconds = static_cast<double>(delay_.count()) / 1000.0;
    return resp;
  }

private:
  std::chrono::milliseconds delay_;
};

/// Failing fake backend: always returns an error.
class FailingFakeInferenceBackend : public InferenceBackend
{
public:
  void initialize() override {}

  InferenceResponse infer(const InferenceRequest & /*request*/) override
  {
    InferenceResponse resp;
    resp.success = false;
    resp.error = "simulated inference failure";
    return resp;
  }
};

}  // namespace cosmos_ros2_video_reasoner
