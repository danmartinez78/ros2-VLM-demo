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

#include "cosmos_ros2_video_reasoner/cosmos_reasoner_node.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "cosmos_ros2_video_reasoner/inference_backend.hpp"

namespace cosmos_ros2_video_reasoner
{

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

CosmosReasonerNode::CosmosReasonerNode(
  std::unique_ptr<InferenceBackend> backend,
  const rclcpp::NodeOptions & options)
: rclcpp::Node("cosmos_reasoner", options),
  backend_(std::move(backend))
{
  declare_parameters();
  validate_parameters();

  // ── initialise backend (throws on failure → node does not finish constructing)
  backend_->initialize();

  // ── result publisher
  source_topic_ = this->get_parameter("image_topic").as_string();
  const std::string result_topic = this->get_parameter("result_topic").as_string();

  if (publish_results_) {
    result_pub_ = this->create_publisher<msg::VisionReasoningResult>(result_topic, 10);
  }

  // ── image subscriber (QoS: best effort, depth 1)
  rclcpp::QoS sub_qos{rclcpp::KeepLast(1)};
  sub_qos.best_effort();
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    source_topic_, sub_qos,
    [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
      image_callback(msg);
    });

  RCLCPP_INFO(this->get_logger(), "Subscribed to %s", source_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing results to %s", result_topic.c_str());

  start_worker();
}

CosmosReasonerNode::~CosmosReasonerNode()
{
  stop_worker();

  RCLCPP_INFO(
    this->get_logger(),
    "Shutdown counters — received: %lu  sampled: %lu  dropped: %lu  "
    "success: %lu  failure: %lu",
    stats_.received, stats_.sampled, stats_.dropped,
    stats_.success, stats_.failure);
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter declarations
// ─────────────────────────────────────────────────────────────────────────────

void CosmosReasonerNode::declare_parameters()
{
  auto desc = [](const std::string & description) {
      rcl_interfaces::msg::ParameterDescriptor d;
      d.description = description;
      return d;
    };

  this->declare_parameter(
    "image_topic", "/camera/image_raw",
    desc("Input image topic (sensor_msgs/msg/Image)"));

  this->declare_parameter(
    "result_topic", "/cosmos/reasoning",
    desc("Output result topic (VisionReasoningResult)"));

  this->declare_parameter(
    "llm_engine_dir", "",
    desc("Absolute path to the LLM TensorRT engine directory"));

  this->declare_parameter(
    "multimodal_engine_dir", "",
    desc("Absolute path to the multimodal (visual encoder) TensorRT engine directory"));

  this->declare_parameter(
    "edge_llm_plugin_path", "",
    desc("Absolute path to libNvInfer_edgellm_plugin.so"));

  this->declare_parameter(
    "prompt",
    "Describe the scene in this camera frame. Identify important objects, "
    "people, animals, vehicles, terrain, hazards, and unusual conditions. "
    "Do not claim details that are not visually supported.",
    desc("Text prompt sent to the VLM for every sampled frame"));

  this->declare_parameter(
    "sample_period_seconds", 2.0,
    desc("Minimum time between frames sent for inference (uses message timestamp)"));

  this->declare_parameter(
    "max_generate_length", 256,
    desc("Maximum number of tokens to generate per frame"));

  this->declare_parameter(
    "temperature", 0.2,
    desc("Sampling temperature"));

  this->declare_parameter(
    "top_p", 0.9,
    desc("Nucleus sampling probability"));

  this->declare_parameter(
    "top_k", 20,
    desc("Top-k sampling parameter"));

  this->declare_parameter(
    "image_max_width", 1280,
    desc("Resize input image to this max width (preserving aspect ratio)"));

  this->declare_parameter(
    "jpeg_quality", 90,
    desc("JPEG quality (1–100) used when encoding frames for the VLM"));

  this->declare_parameter(
    "queue_capacity", 1,
    desc("Depth of the inference queue (must be 1; future expansion)"));

  this->declare_parameter(
    "drop_old_frames", true,
    desc("When the worker is busy, replace the pending frame with the newest one"));

  this->declare_parameter(
    "publish_results", true,
    desc("Publish VisionReasoningResult messages (disable for benchmark-only mode)"));

  this->declare_parameter(
    "dump_profile", false,
    desc("Enable TensorRT Edge-LLM profiling output"));

  this->declare_parameter(
    "profile_output_directory", "/tmp/cosmos_ros2_profiles",
    desc("Directory for profiling output files"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter validation / caching
// ─────────────────────────────────────────────────────────────────────────────

void CosmosReasonerNode::validate_parameters()
{
  sample_period_seconds_ = this->get_parameter("sample_period_seconds").as_double();
  if (sample_period_seconds_ <= 0.0) {
    throw std::runtime_error("sample_period_seconds must be > 0");
  }

  max_generate_length_ = this->get_parameter("max_generate_length").as_int();
  if (max_generate_length_ <= 0) {
    throw std::runtime_error("max_generate_length must be > 0");
  }

  temperature_ = static_cast<float>(this->get_parameter("temperature").as_double());
  if (temperature_ < 0.0f) {
    throw std::runtime_error("temperature must be >= 0");
  }

  top_p_ = static_cast<float>(this->get_parameter("top_p").as_double());
  if (top_p_ <= 0.0f || top_p_ > 1.0f) {
    throw std::runtime_error("top_p must be in (0, 1]");
  }

  top_k_ = this->get_parameter("top_k").as_int();
  if (top_k_ <= 0) {
    throw std::runtime_error("top_k must be > 0");
  }

  image_max_width_ = this->get_parameter("image_max_width").as_int();
  if (image_max_width_ <= 0) {
    throw std::runtime_error("image_max_width must be > 0");
  }

  jpeg_quality_ = this->get_parameter("jpeg_quality").as_int();
  if (jpeg_quality_ < 1 || jpeg_quality_ > 100) {
    throw std::runtime_error("jpeg_quality must be in [1, 100]");
  }

  prompt_ = this->get_parameter("prompt").as_string();
  if (prompt_.empty()) {
    throw std::runtime_error("prompt must not be empty");
  }

  const auto queue_capacity = this->get_parameter("queue_capacity").as_int();
  if (queue_capacity != 1) {
    throw std::runtime_error("queue_capacity must be 1 in the current implementation");
  }

  drop_old_frames_ = this->get_parameter("drop_old_frames").as_bool();
  publish_results_ = this->get_parameter("publish_results").as_bool();
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker thread
// ─────────────────────────────────────────────────────────────────────────────

void CosmosReasonerNode::start_worker()
{
  worker_running_ = true;
  worker_thread_ = std::thread(&CosmosReasonerNode::worker_loop, this);
}

void CosmosReasonerNode::stop_worker()
{
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    worker_running_ = false;
  }
  queue_cv_.notify_all();

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

void CosmosReasonerNode::worker_loop()
{
  while (true) {
    PendingFrame frame;

    // ── wait for a frame ─────────────────────────────────────────────────
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this] {
        return pending_frame_.has_value() || !worker_running_;
      });

      if (!worker_running_ && !pending_frame_.has_value()) {
        break;
      }
      if (!pending_frame_.has_value()) {
        break;
      }
      frame = *pending_frame_;
      pending_frame_.reset();
    }

    // ── convert image ─────────────────────────────────────────────────────
    cv::Mat bgr;
    try {
      auto cv_img = cv_bridge::toCvShare(frame.msg, "bgr8");
      bgr = cv_img->image.clone();
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed: %s", e.what());
      ++stats_.failure;
      InferenceResponse resp;
      resp.success = false;
      resp.error = std::string("cv_bridge: ") + e.what();
      publish_result(frame.msg->header, frame.seq, resp);
      continue;
    }

    // ── resize if wider than image_max_width ─────────────────────────────
    if (bgr.cols > image_max_width_) {
      const double scale = static_cast<double>(image_max_width_) / bgr.cols;
      const int new_h = static_cast<int>(bgr.rows * scale);
      cv::resize(bgr, bgr, cv::Size(image_max_width_, new_h), 0.0, 0.0, cv::INTER_AREA);
    }

    // ── build and run inference request ──────────────────────────────────
    InferenceRequest req;
    req.image = bgr;
    req.prompt = prompt_;
    req.max_generate_length = max_generate_length_;
    req.temperature = temperature_;
    req.top_p = top_p_;
    req.top_k = top_k_;

    InferenceResponse resp;
    try {
      resp = backend_->infer(req);
    } catch (const std::exception & e) {
      resp.success = false;
      resp.error = std::string("backend exception: ") + e.what();
    }

    if (resp.success) {
      ++stats_.success;
      RCLCPP_INFO(
        this->get_logger(),
        "[frame %lu | %.3f s] %s",
        frame.seq, resp.inference_seconds, resp.text.c_str());
    } else {
      ++stats_.failure;
      RCLCPP_ERROR(
        this->get_logger(),
        "[frame %lu] inference failed: %s",
        frame.seq, resp.error.c_str());
    }

    publish_result(frame.msg->header, frame.seq, resp);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Image subscription callback
// ─────────────────────────────────────────────────────────────────────────────

void CosmosReasonerNode::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  ++stats_.received;

  // ── timestamp-based sampling (deterministic with bag replay) ─────────
  const rclcpp::Time msg_time(msg->header.stamp, RCL_ROS_TIME);
  if (have_last_time_) {
    const double elapsed = (msg_time - last_sampled_time_).seconds();
    if (elapsed < sample_period_seconds_) {
      return;  // not yet due
    }
  }

  last_sampled_time_ = msg_time;
  have_last_time_ = true;
  ++stats_.sampled;

  // ── enqueue (bounded depth 1) ─────────────────────────────────────────
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (pending_frame_.has_value() && drop_old_frames_) {
      ++stats_.dropped;
    }
    pending_frame_ = PendingFrame{msg, stats_.sampled};
  }
  queue_cv_.notify_one();
}

// ─────────────────────────────────────────────────────────────────────────────
// Result publication
// ─────────────────────────────────────────────────────────────────────────────

void CosmosReasonerNode::publish_result(
  const std_msgs::msg::Header & header,
  uint64_t frame_seq,
  const InferenceResponse & resp)
{
  if (!publish_results_ || !result_pub_) {
    return;
  }

  msg::VisionReasoningResult out;
  out.header = header;
  out.source_topic = source_topic_;
  out.prompt = prompt_;
  out.response = resp.text;
  out.inference_seconds = resp.inference_seconds;
  out.frame_sequence = frame_seq;
  out.success = resp.success;
  out.error = resp.error;

  result_pub_->publish(out);
}

}  // namespace cosmos_ros2_video_reasoner
