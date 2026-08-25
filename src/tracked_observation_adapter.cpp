// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

#include "edge_vlm_ros/tracked_observation_adapter.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace edge_vlm_ros
{

namespace
{

TrackedBox detection_to_tracked_box(const vision_msgs::msg::Detection2D & detection)
{
  TrackedBox box;
  box.center_x = static_cast<float>(detection.bbox.center.position.x);
  box.center_y = static_cast<float>(detection.bbox.center.position.y);
  box.width = static_cast<float>(detection.bbox.size_x);
  box.height = static_cast<float>(detection.bbox.size_y);

  if (!detection.results.empty()) {
    const auto & hypothesis = detection.results.front();
    box.class_label = hypothesis.hypothesis.class_id;
    box.confidence = static_cast<float>(hypothesis.hypothesis.score);
  }
  return box;
}

msg::TrackedObject track_state_to_msg(
  const std_msgs::msg::Header & header,
  const TrackState & track)
{
  msg::TrackedObject out;
  out.header = header;
  out.track_id = track.track_id;
  out.class_label = track.box.class_label;
  out.confidence = track.box.confidence;
  out.center_x = track.box.center_x;
  out.center_y = track.box.center_y;
  out.width = track.box.width;
  out.height = track.box.height;
  out.velocity_x = track.velocity_x;
  out.velocity_y = track.velocity_y;
  out.age = track.age;
  out.coast_age = track.coast_age;
  return out;
}

bool same_stamp(const builtin_interfaces::msg::Time & lhs, const builtin_interfaces::msg::Time & rhs)
{
  return lhs.sec == rhs.sec && lhs.nanosec == rhs.nanosec;
}

}  // namespace

TrackedObservationAdapter::TrackedObservationAdapter(const rclcpp::NodeOptions & options)
: Node("tracked_observation_adapter", options)
{
  declare_parameters();
  validate_parameters();

  tracker_ = IouTracker(IouTracker::Options{
    tracker_min_iou_,
    static_cast<uint32_t>(tracker_max_coast_age_),
    tracker_class_aware_});

  tracked_observation_pub_ = this->create_publisher<msg::TrackedObservation>(
    tracked_observation_topic_, rclcpp::SystemDefaultsQoS());

  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    image_topic_, rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
      image_callback(msg);
    });
  detections_sub_ = this->create_subscription<DetectionArray>(
    detections_topic_, rclcpp::SensorDataQoS(),
    [this](const DetectionArray::ConstSharedPtr msg) {
      detections_callback(msg);
    });
}

void TrackedObservationAdapter::declare_parameters()
{
  const auto desc = [](const std::string & text) {
      rcl_interfaces::msg::ParameterDescriptor d;
      d.description = text;
      return d;
    };

  this->declare_parameter(
    "image_topic", "/camera0/color/image_raw",
    desc("Source image topic synchronized with detections"));
  this->declare_parameter(
    "detections_topic", "/detections",
    desc("Detector output topic (vision_msgs/msg/Detection2DArray)"));
  this->declare_parameter(
    "tracked_observation_topic", "/tracked_observation",
    desc("Output tracked-observation topic"));
  this->declare_parameter(
    "detector_id", "isaac_ros_rtdetr",
    desc("Detector identity stamped into tracked observations"));
  this->declare_parameter(
    "tracker_id", "iou_tracker",
    desc("Tracker identity stamped into tracked observations"));
  this->declare_parameter(
    "tracker_min_iou", 0.3,
    desc("Minimum IOU for associating a detection to an existing track"));
  this->declare_parameter(
    "tracker_max_coast_age", 1,
    desc("How many unmatched updates a track may coast before expiring"));
  this->declare_parameter(
    "tracker_class_aware", true,
    desc("Require matching class labels when associating detections to tracks"));
}

void TrackedObservationAdapter::validate_parameters()
{
  image_topic_ = this->get_parameter("image_topic").as_string();
  detections_topic_ = this->get_parameter("detections_topic").as_string();
  tracked_observation_topic_ = this->get_parameter("tracked_observation_topic").as_string();
  detector_id_ = this->get_parameter("detector_id").as_string();
  tracker_id_ = this->get_parameter("tracker_id").as_string();
  tracker_min_iou_ = static_cast<float>(this->get_parameter("tracker_min_iou").as_double());
  tracker_max_coast_age_ = this->get_parameter("tracker_max_coast_age").as_int();
  tracker_class_aware_ = this->get_parameter("tracker_class_aware").as_bool();

  if (image_topic_.empty() || detections_topic_.empty() || tracked_observation_topic_.empty()) {
    throw std::runtime_error("TrackedObservationAdapter topics must be non-empty");
  }
  if (tracker_min_iou_ < 0.0f || tracker_min_iou_ > 1.0f) {
    throw std::runtime_error("tracker_min_iou must be in [0, 1]");
  }
  if (tracker_max_coast_age_ < 0) {
    throw std::runtime_error("tracker_max_coast_age must be >= 0");
  }
}

void TrackedObservationAdapter::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & image_msg)
{
  if (latest_image_.has_value()) {
    ++stale_drop_count_;
  }
  latest_image_ = *image_msg;
  try_publish_latest_match();
}

void TrackedObservationAdapter::detections_callback(const DetectionArray::ConstSharedPtr & detections_msg)
{
  if (latest_detections_.has_value()) {
    ++stale_drop_count_;
  }
  latest_detections_ = *detections_msg;
  try_publish_latest_match();
}

void TrackedObservationAdapter::try_publish_latest_match()
{
  if (!latest_image_.has_value() || !latest_detections_.has_value()) {
    return;
  }

  const auto & image = *latest_image_;
  const auto & detections = *latest_detections_;
  if (!same_stamp(image.header.stamp, detections.header.stamp)) {
    if (rclcpp::Time(image.header.stamp, RCL_ROS_TIME) < rclcpp::Time(detections.header.stamp, RCL_ROS_TIME)) {
      latest_image_.reset();
    } else {
      latest_detections_.reset();
    }
    ++mismatch_drop_count_;
    return;
  }

  const rclcpp::Time completed_at = this->now();
  auto observation = build_observation(image, detections, completed_at);
  tracked_observation_pub_->publish(observation);
  ++published_count_;
  latest_image_.reset();
  latest_detections_.reset();
}

msg::TrackedObservation TrackedObservationAdapter::build_observation(
  const sensor_msgs::msg::Image & image_msg,
  const DetectionArray & detections_msg,
  const rclcpp::Time & completed_at)
{
  std::vector<TrackedBox> detections;
  detections.reserve(detections_msg.detections.size());
  for (const auto & detection : detections_msg.detections) {
    detections.push_back(detection_to_tracked_box(detection));
  }

  const std::vector<TrackState> tracks = tracker_.update(detections);

  msg::TrackedObservation out;
  out.header = image_msg.header;
  out.source_image = image_msg;
  out.source_stamp = image_msg.header.stamp;
  out.source_topic = image_topic_;
  out.detector_completed_at = detections_msg.header.stamp;
  out.tracker_completed_at = completed_at;
  out.detector_id = detector_id_;
  out.tracker_id = tracker_id_;
  out.source_sequence = next_source_sequence_++;

  out.tracked_objects.reserve(tracks.size());
  for (const auto & track : tracks) {
    out.tracked_objects.push_back(track_state_to_msg(image_msg.header, track));
  }
  return out;
}

}  // namespace edge_vlm_ros
