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

#include "cosmos_ros2_video_reasoner/tensorrt_edge_llm_backend.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char ** argv)
{
  if (argc != 5) {
    std::cerr
      << "Usage: " << argv[0]
      << " <llm-engine-dir> <multimodal-engine-dir> <plugin-path> <image>\n";
    return 2;
  }

  auto run_backend = [&]() -> int {
    try {
      cosmos_ros2_video_reasoner::TensorRTEdgeLLMConfig config;
    config.llm_engine_dir = argv[1];
    config.multimodal_engine_dir = argv[2];
    config.edge_llm_plugin_path = argv[3];
    config.jpeg_quality = 90;

    cv::Mat image = cv::imread(argv[4], cv::IMREAD_COLOR);
    if (image.empty()) {
      std::cerr << "Could not read image: " << argv[4] << '\n';
      return 3;
    }

    std::cout << "Image: " << image.cols << "x" << image.rows << '\n';
    std::cout << "Initializing direct-linked backend...\n";

    cosmos_ros2_video_reasoner::TensorRTEdgeLLMBackend backend(config);
    backend.initialize();

    cosmos_ros2_video_reasoner::InferenceRequest request;
    request.image = image;
    request.prompt = "Describe what you see in this image.";
    request.max_generate_length = 64;
    request.temperature = 1.0F;
    request.top_p = 1.0F;
    request.top_k = 50;

    std::cout << "Backend initialized. Starting inference...\n";
    auto const response = backend.infer(request);

    if (!response.success) {
      std::cerr << "Inference failed: " << response.error << '\n';
      return 4;
    }

    std::cout << "Inference time: " << response.inference_seconds << " seconds\n";
    std::cout << "Response: " << response.text << '\n';
      return 0;
    } catch (std::exception const & error) {
      std::cerr << "Exception: " << error.what() << '\n';
      return 5;
    }
  };

  if (std::getenv("COSMOS_SMOKE_THREADED") == nullptr) {
    std::cout << "Execution mode: main thread\n";
    return run_backend();
  }

  std::cout << "Execution mode: std::thread worker\n";
  int result = 5;
  std::thread worker([&]() {result = run_backend();});
  worker.join();
  return result;
}
