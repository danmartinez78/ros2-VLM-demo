// Copyright 2025 edge_vlm_ros contributors
#pragma once

#include "edge_vlm_ros/inference_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace edge_vlm_ros
{
struct IpcInferenceConfig
{
  std::string socket_path{"/tmp/edge_vlm.sock"};
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
}  // namespace edge_vlm_ros
