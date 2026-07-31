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

#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QSize>
#include <QTextEdit>
#include <QTimer>
#include <QWidget>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "cosmos_ros2_video_reasoner/msg/vision_reasoning_result.hpp"

namespace cosmos_ros2_video_reasoner
{

class VisionReasoningPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit VisionReasoningPanel(QWidget * parent = nullptr);
  ~VisionReasoningPanel() override;

private Q_SLOTS:
  void refresh_display();

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void result_callback(const msg::VisionReasoningResult::ConstSharedPtr & msg);

  static std::optional<QImage> to_qimage(const sensor_msgs::msg::Image & msg);
  std::optional<QImage> find_cached_image_for_stamp(const builtin_interfaces::msg::Time & stamp) const;

  QLabel * status_label_{nullptr};
  QLabel * image_label_{nullptr};
  QLineEdit * result_stamp_line_{nullptr};
  QLineEdit * image_stamp_line_{nullptr};
  QLineEdit * frame_sequence_line_{nullptr};
  QLineEdit * latency_line_{nullptr};
  QTextEdit * prompt_text_{nullptr};
  QTextEdit * response_text_{nullptr};
  QTextEdit * error_text_{nullptr};

  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<msg::VisionReasoningResult>::SharedPtr result_sub_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread spin_thread_;
  QTimer * refresh_timer_{nullptr};

  mutable std::mutex data_mutex_;
  std::optional<msg::VisionReasoningResult> latest_result_;
  std::optional<rclcpp::Time> latest_result_received_at_;
  std::optional<rclcpp::Time> latest_image_stamp_;
  std::deque<std::pair<rclcpp::Time, QImage>> image_cache_;
  static constexpr size_t kImageCacheLimit = 32;

  std::optional<rclcpp::Time> displayed_result_stamp_;
  QSize displayed_label_size_;
};

}  // namespace cosmos_ros2_video_reasoner
