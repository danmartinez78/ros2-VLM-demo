// Copyright 2025 cosmos_ros2_video_reasoner contributors
#pragma once

#include "cosmos_ros2_video_reasoner/inference_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace cosmos_ros2_video_reasoner
{
struct IpcInferenceConfig
{
  std::string socket_path{"/tmp/cosmos_edge_llm.sock"};
  int connect_timeout_seconds{120};
  int request_timeout_seconds{90};
  size_t max_image_bytes{256U * 1024U * 1024U};
  size_t max_text_bytes{1024U * 1024U};
};

class IpcInferenceBackend : public InferenceBackend
{
public:
  explicit IpcInferenceBackend(IpcInferenceConfig config);
  ~IpcInferenceBackend() override;
  void initialize() override;
  InferenceResponse infer(InferenceRequest const & request) override;

private:
  void connect_worker();
  void close_connection() noexcept;

  IpcInferenceConfig config_;
  int socket_fd_{-1};
  uint64_t next_request_id_{1};
};
}  // namespace cosmos_ros2_video_reasoner
