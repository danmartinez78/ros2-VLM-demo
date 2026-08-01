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

#include "edge_vlm_ros/vision_reasoning_rviz_panel.hpp"

#include <algorithm>
#include <functional>
#include <optional>

#include <QFormLayout>
#include <QHBoxLayout>
#include <QPixmap>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>

#include "edge_vlm_ros/vision_reasoning_rviz_formatting.hpp"

namespace edge_vlm_ros
{

VisionReasoningPanel::VisionReasoningPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * main_layout = new QVBoxLayout();

  status_label_ = new QLabel("NO RESULT");
  status_label_->setStyleSheet("QLabel { background: #777777; color: white; font-weight: bold; padding: 4px; }");
  main_layout->addWidget(status_label_);

  image_label_ = new QLabel("No matching image for result stamp yet");
  image_label_->setMinimumHeight(220);
  image_label_->setAlignment(Qt::AlignCenter);
  image_label_->setStyleSheet("QLabel { border: 1px solid #666666; }");
  main_layout->addWidget(image_label_);

  auto * details_layout = new QFormLayout();
  result_stamp_line_ = new QLineEdit();
  result_stamp_line_->setReadOnly(true);
  image_stamp_line_ = new QLineEdit();
  image_stamp_line_->setReadOnly(true);
  frame_sequence_line_ = new QLineEdit();
  frame_sequence_line_->setReadOnly(true);
  latency_line_ = new QLineEdit();
  latency_line_->setReadOnly(true);

  details_layout->addRow("Result stamp", result_stamp_line_);
  details_layout->addRow("Latest image stamp", image_stamp_line_);
  details_layout->addRow("Frame sequence", frame_sequence_line_);
  details_layout->addRow("Inference latency (s)", latency_line_);
  main_layout->addLayout(details_layout);

  prompt_text_ = new QTextEdit();
  prompt_text_->setReadOnly(true);
  response_text_ = new QTextEdit();
  response_text_->setReadOnly(true);
  error_text_ = new QTextEdit();
  error_text_->setReadOnly(true);

  auto * text_layout = new QHBoxLayout();
  auto * prompt_layout = new QVBoxLayout();
  prompt_layout->addWidget(new QLabel("Prompt"));
  prompt_layout->addWidget(prompt_text_);
  auto * response_layout = new QVBoxLayout();
  response_layout->addWidget(new QLabel("Response"));
  response_layout->addWidget(response_text_);
  text_layout->addLayout(prompt_layout);
  text_layout->addLayout(response_layout);
  main_layout->addLayout(text_layout);

  main_layout->addWidget(new QLabel("Error / details"));
  main_layout->addWidget(error_text_);

  setLayout(main_layout);

  node_ = std::make_shared<rclcpp::Node>("vision_reasoning_panel");
  rclcpp::QoS image_qos{rclcpp::KeepLast(10)};
  image_qos.best_effort();
  image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
    "/camera/image_raw", image_qos,
    std::bind(&VisionReasoningPanel::image_callback, this, std::placeholders::_1));

  result_sub_ = node_->create_subscription<msg::VlmResult>(
    "/vlm/result", rclcpp::SystemDefaultsQoS(),
    std::bind(&VisionReasoningPanel::result_callback, this, std::placeholders::_1));

  executor_.add_node(node_);
  spin_thread_ = std::thread([this]() {executor_.spin();});

  refresh_timer_ = new QTimer(this);
  connect(refresh_timer_, &QTimer::timeout, this, &VisionReasoningPanel::refresh_display);
  refresh_timer_->start(250);
}

VisionReasoningPanel::~VisionReasoningPanel()
{
  if (refresh_timer_) {
    refresh_timer_->stop();
  }
  executor_.cancel();
  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }
  executor_.remove_node(node_);
}

void VisionReasoningPanel::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto image = to_qimage(*msg);
  if (!image.has_value()) {
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  const rclcpp::Time stamp(msg->header.stamp, RCL_ROS_TIME);
  latest_image_stamp_ = stamp;
  image_cache_.emplace_back(stamp, image.value());
  while (image_cache_.size() > kImageCacheLimit) {
    image_cache_.pop_front();
  }
}

void VisionReasoningPanel::result_callback(const msg::VlmResult::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_result_ = *msg;
  latest_result_received_at_ = node_->now();
}

void VisionReasoningPanel::refresh_display()
{
  std::optional<msg::VlmResult> latest_result;
  std::optional<rclcpp::Time> latest_image_stamp;
  std::optional<rclcpp::Time> result_received_at;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_result = latest_result_;
    latest_image_stamp = latest_image_stamp_;
    result_received_at = latest_result_received_at_;
  }

  const rclcpp::Time now = node_->now();
  const auto presentation = rviz::build_result_presentation(
    latest_result ? &latest_result.value() : nullptr,
    latest_image_stamp,
    now,
    2.5);

  if (presentation.state == rviz::ResultState::kSuccess) {
    status_label_->setStyleSheet("QLabel { background: #2e7d32; color: white; font-weight: bold; padding: 4px; }");
  } else if (presentation.state == rviz::ResultState::kStale) {
    status_label_->setStyleSheet("QLabel { background: #ef6c00; color: white; font-weight: bold; padding: 4px; }");
  } else if (presentation.state == rviz::ResultState::kFailed) {
    status_label_->setStyleSheet("QLabel { background: #c62828; color: white; font-weight: bold; padding: 4px; }");
  } else {
    status_label_->setStyleSheet("QLabel { background: #777777; color: white; font-weight: bold; padding: 4px; }");
  }
  status_label_->setText(QString::fromStdString(presentation.status_text));

  if (!latest_result.has_value()) {
    result_stamp_line_->setText("");
    frame_sequence_line_->setText("");
    latency_line_->setText("");
    prompt_text_->setPlainText("");
    response_text_->setPlainText("");
    error_text_->setPlainText(QString::fromStdString(presentation.details_text));
    image_label_->setText("Waiting for VlmResult");
    return;
  }

  result_stamp_line_->setText(QString::fromStdString(rviz::format_stamp(latest_result->header.stamp)));
  if (latest_image_stamp.has_value()) {
    image_stamp_line_->setText(QString::fromStdString(rviz::format_stamp(latest_image_stamp.value())));
  } else {
    image_stamp_line_->setText("");
  }

  frame_sequence_line_->setText(QString::number(latest_result->frame_sequence));
  latency_line_->setText(QString::number(latest_result->inference_seconds, 'f', 3));
  prompt_text_->setPlainText(QString::fromStdString(latest_result->prompt));
  response_text_->setPlainText(QString::fromStdString(latest_result->response));

  QString details = QString::fromStdString(presentation.details_text);
  if (!latest_result->success && !latest_result->error.empty()) {
    details.append("\n").append(QString::fromStdString(latest_result->error));
  }
  if (result_received_at.has_value()) {
    details.append("\nreceived_at=").append(QString::number(result_received_at->seconds(), 'f', 6));
  }
  error_text_->setPlainText(details);

  auto maybe_image = find_cached_image_for_stamp(latest_result->header.stamp);
  if (maybe_image.has_value()) {
    const rclcpp::Time result_stamp(latest_result->header.stamp, RCL_ROS_TIME);
    const QSize current_size = image_label_->size();
    const bool stamp_changed = !displayed_result_stamp_.has_value() ||
      displayed_result_stamp_.value() != result_stamp;
    const bool size_changed = current_size != displayed_label_size_;

    if (stamp_changed || size_changed) {
      image_label_->setPixmap(QPixmap::fromImage(maybe_image.value()).scaled(
        current_size, Qt::KeepAspectRatio, Qt::SmoothTransformation));
      displayed_result_stamp_ = result_stamp;
      displayed_label_size_ = current_size;
    }
    image_label_->setText("");
  } else {
    displayed_result_stamp_ = std::nullopt;
    image_label_->setPixmap(QPixmap());
    image_label_->setText("No cached image for result stamp");
  }
}

std::optional<QImage> VisionReasoningPanel::to_qimage(const sensor_msgs::msg::Image & msg)
{
  const int width = static_cast<int>(msg.width);
  const int height = static_cast<int>(msg.height);
  if (width <= 0 || height <= 0) {
    return std::nullopt;
  }
  const size_t step = static_cast<size_t>(msg.step);
  if (step == 0 || msg.data.size() < step * static_cast<size_t>(height)) {
    return std::nullopt;
  }

  if (msg.encoding == "rgb8") {
    if (step < static_cast<size_t>(width) * 3u) {
      return std::nullopt;
    }
    QImage image(msg.data.data(), width, height, static_cast<int>(step), QImage::Format_RGB888);
    return image.copy();
  }

  if (msg.encoding == "bgr8") {
    if (step < static_cast<size_t>(width) * 3u) {
      return std::nullopt;
    }
    QImage bgr(msg.data.data(), width, height, static_cast<int>(step), QImage::Format_BGR888);
    return bgr.copy();
  }

  if (msg.encoding == "mono8") {
    if (step < static_cast<size_t>(width)) {
      return std::nullopt;
    }
    QImage mono(msg.data.data(), width, height, static_cast<int>(step), QImage::Format_Grayscale8);
    return mono.copy();
  }

  return std::nullopt;
}

std::optional<QImage> VisionReasoningPanel::find_cached_image_for_stamp(
  const builtin_interfaces::msg::Time & stamp) const
{
  const rclcpp::Time target(stamp, RCL_ROS_TIME);
  std::lock_guard<std::mutex> lock(data_mutex_);
  auto it = std::find_if(
    image_cache_.rbegin(), image_cache_.rend(),
    [&target](const auto & item) {
      return item.first == target;
    });

  if (it == image_cache_.rend()) {
    return std::nullopt;
  }
  return it->second;
}

}  // namespace edge_vlm_ros

PLUGINLIB_EXPORT_CLASS(edge_vlm_ros::VisionReasoningPanel, rviz_common::Panel)
