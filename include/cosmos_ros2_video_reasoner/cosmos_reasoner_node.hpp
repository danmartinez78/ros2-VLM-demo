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
#include "rclcpp/rclcpp.hpp"
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "cosmos_ros2_video_reasoner/msg/vision_reasoning_result.hpp"

namespace cosmos_ros2_video_reasoner
{

/// ROS 2 node that samples camera frames, sends them to a VLM backend,
/// and publishes inference results.
///
/// Threading model
/// ───────────────
///  • ROS executor thread  – runs the image subscription callback; must return quickly.
///  • Inference worker thread – dequeued from the bounded frame queue; calls backend.infer().
///
/// The two threads share a single pending frame slot protected by a mutex + condition variable.
/// When `drop_old_frames` is true and a new frame arrives before the worker finishes the
/// previous one, the older frame is replaced.
class CosmosReasonerNode : public rclcpp::Node
{
public:
  explicit CosmosReasonerNode(
    std::unique_ptr<InferenceBackend> backend,
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions{});

  ~CosmosReasonerNode() override;

  // ── Counters (primarily for unit testing and shutdown logging) ──────────
  uint64_t received_count() const noexcept {return stats_.received;}
  uint64_t sampled_count() const noexcept {return stats_.sampled;}
  uint64_t dropped_count() const noexcept {return stats_.dropped;}
  uint64_t success_count() const noexcept {return stats_.success;}
  uint64_t failure_count() const noexcept {return stats_.failure;}

private:
  // ── Parameter declarations ──────────────────────────────────────────────
  void declare_parameters();

  // ── Startup validation ──────────────────────────────────────────────────
  void validate_parameters();

  // ── Worker thread lifecycle ─────────────────────────────────────────────
  void start_worker();
  void stop_worker();
  void worker_loop();

  // ── Subscription callback (fast; must not block) ────────────────────────
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  // ── Result publication ───────────────────────────────────────────────────
  void publish_result(
    const std_msgs::msg::Header & header,
    uint64_t frame_seq,
    const InferenceResponse & resp,
    const std::string & effective_prompt);

  // ── Prompt/history configuration helpers ─────────────────────────────────
  std::string render_effective_prompt(
    uint64_t frame_seq,
    bool suppress_system_and_context = false) const;
  void validate_template_variables(const std::string & name, const std::string & templ) const;
  void maybe_reset_observation_history_before_request();
  void update_observation_history_after_response(
    const InferenceResponse & resp, const std::string & user_text);
  size_t observation_history_size_chars() const;

  // ── Backend ─────────────────────────────────────────────────────────────
  std::unique_ptr<InferenceBackend> backend_;

  // ── ROS interfaces ──────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<msg::VisionReasoningResult>::SharedPtr result_pub_;

  // ── Sampling state ──────────────────────────────────────────────────────
  rclcpp::Time last_sampled_time_{0, 0, RCL_ROS_TIME};
  double sample_period_seconds_{2.0};
  bool have_last_time_{false};

  // ── Parameters (cached after validate_parameters) ───────────────────────
  std::string source_topic_;
  std::string legacy_prompt_;
  std::string task_profile_;
  std::string prompt_version_;
  std::string prompt_config_hash_;
  std::string system_instruction_;
  std::string task_instruction_;
  std::string instruction_delivery_mode_{"inline"};
  std::string active_prompt_template_;
  int observation_history_max_entries_{0};
  int observation_history_max_chars_{0};
  std::string observation_history_reset_policy_{"never"};
  int observation_history_reset_interval_requests_{0};
  std::unordered_map<std::string, std::string> task_profiles_;
  int max_generate_length_{256};
  float temperature_{0.2f};
  float top_p_{0.9f};
  int top_k_{20};
  int image_max_width_{1280};
  int jpeg_quality_{90};
  bool drop_old_frames_{true};
  bool publish_results_{true};
  bool enable_system_prompt_cache_{false};

  // ── Worker thread and synchronisation ───────────────────────────────────
  std::thread worker_thread_;
  bool worker_running_{false};
  bool backend_init_complete_{false};
  std::exception_ptr backend_init_error_;
  std::condition_variable backend_init_cv_;

  struct PendingFrame
  {
    sensor_msgs::msg::Image::ConstSharedPtr msg;
    uint64_t seq{0};
    int64_t subscribe_wall_ns{0};  // wall time when frame was accepted in image_callback
  };

  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::optional<PendingFrame> pending_frame_;  // bounded queue of depth 1
  std::deque<HistoryEntry> observation_history_;
  uint64_t requests_since_observation_history_reset_{0};

  // ── Counters ─────────────────────────────────────────────────────────────
  struct Stats
  {
    uint64_t received{0};
    uint64_t sampled{0};
    uint64_t dropped{0};
    uint64_t success{0};
    uint64_t failure{0};
  } stats_;

  // ── Benchmark output (optional) ───────────────────────────────────────────
  // When benchmark_output_file is non-empty, per-frame timing JSON lines are
  // written to this file. Disabled by default; no runtime overhead when empty.
  std::string benchmark_output_file_;
  std::unique_ptr<std::ofstream> benchmark_out_;
  int64_t node_init_wall_ns_{0};    // wall time captured at the start of the constructor
  int64_t worker_ready_wall_ns_{0}; // wall time when inference worker finished initialising
};

}  // namespace cosmos_ros2_video_reasoner
