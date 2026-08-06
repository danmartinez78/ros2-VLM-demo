// Copyright 2025 edge_vlm_ros contributors

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include "edge_vlm_ros/msg/tracked_observation.hpp"
#include "edge_vlm_ros/tracked_observation_adapter.hpp"

using namespace std::chrono_literals;

namespace
{

sensor_msgs::msg::Image make_image(int sec, uint8_t pixel)
{
  sensor_msgs::msg::Image img;
  img.header.stamp = rclcpp::Time(sec, 0, RCL_ROS_TIME);
  img.header.frame_id = "camera";
  img.encoding = "bgr8";
  img.width = 2;
  img.height = 1;
  img.step = 6;
  img.data.assign(6, pixel);
  return img;
}

vision_msgs::msg::Detection2DArray make_detections(int sec, double x, const std::string & class_id)
{
  vision_msgs::msg::Detection2DArray out;
  out.header.stamp = rclcpp::Time(sec, 0, RCL_ROS_TIME);

  vision_msgs::msg::Detection2D detection;
  detection.bbox.center.position.x = x;
  detection.bbox.center.position.y = 4.0;
  detection.bbox.size_x = 2.0;
  detection.bbox.size_y = 2.0;

  vision_msgs::msg::ObjectHypothesisWithPose result;
  result.hypothesis.class_id = class_id;
  result.hypothesis.score = 0.9;
  detection.results.push_back(result);
  out.detections.push_back(detection);
  return out;
}

bool spin_until(
  const rclcpp::Node::SharedPtr & n1,
  const rclcpp::Node::SharedPtr & n2,
  const std::function<bool()> & pred,
  std::chrono::milliseconds timeout = 2s)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    rclcpp::spin_some(n1);
    rclcpp::spin_some(n2);
    if (pred()) {
      return true;
    }
    std::this_thread::sleep_for(10ms);
  }
  return pred();
}

}  // namespace

TEST(TrackedObservationAdapter, PublishesExactImageDetectionPairing)
{
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("image_topic", "/camera/image_raw");
  opts.append_parameter_override("detections_topic", "/detections");
  opts.append_parameter_override("tracked_observation_topic", "/tracked_observation");
  auto adapter = std::make_shared<edge_vlm_ros::TrackedObservationAdapter>(opts);
  auto helper = std::make_shared<rclcpp::Node>("tracked_observation_adapter_test_helper");

  edge_vlm_ros::msg::TrackedObservation observed;
  bool received = false;
  auto sub = helper->create_subscription<edge_vlm_ros::msg::TrackedObservation>(
    "/tracked_observation", rclcpp::SystemDefaultsQoS(),
    [&](edge_vlm_ros::msg::TrackedObservation::SharedPtr msg) {
      observed = *msg;
      received = true;
    });

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto image_pub = helper->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  auto detections_pub = helper->create_publisher<vision_msgs::msg::Detection2DArray>("/detections", qos);

  std::this_thread::sleep_for(100ms);
  const auto image = make_image(42, 123);
  const auto detections = make_detections(42, 11.0, "person");
  image_pub->publish(image);
  detections_pub->publish(detections);

  ASSERT_TRUE(spin_until(adapter, helper, [&] {return received;}));
  EXPECT_EQ(observed.source_image.header.stamp.sec, image.header.stamp.sec);
  EXPECT_EQ(observed.source_image.data, image.data);
  ASSERT_EQ(observed.tracked_objects.size(), 1u);
  EXPECT_EQ(observed.tracked_objects.front().class_label, "person");
  EXPECT_FLOAT_EQ(observed.tracked_objects.front().center_x, 11.0f);
  EXPECT_EQ(observed.source_sequence, 1u);
}


TEST(TrackedObservationAdapter, DoesNotPairMismatchedTimestamps)
{
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("image_topic", "/camera/image_raw");
  opts.append_parameter_override("detections_topic", "/detections");
  opts.append_parameter_override("tracked_observation_topic", "/tracked_observation");
  auto adapter = std::make_shared<edge_vlm_ros::TrackedObservationAdapter>(opts);
  auto helper = std::make_shared<rclcpp::Node>("tracked_observation_adapter_mismatch_test_helper");

  std::vector<edge_vlm_ros::msg::TrackedObservation> observations;
  auto sub = helper->create_subscription<edge_vlm_ros::msg::TrackedObservation>(
    "/tracked_observation", rclcpp::SystemDefaultsQoS(),
    [&](edge_vlm_ros::msg::TrackedObservation::SharedPtr msg) {
      observations.push_back(*msg);
    });

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto image_pub = helper->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  auto detections_pub = helper->create_publisher<vision_msgs::msg::Detection2DArray>("/detections", qos);

  std::this_thread::sleep_for(100ms);
  image_pub->publish(make_image(10, 10));
  detections_pub->publish(make_detections(11, 9.0, "person"));
  std::this_thread::sleep_for(200ms);
  rclcpp::spin_some(adapter);
  rclcpp::spin_some(helper);
  EXPECT_TRUE(observations.empty());

  image_pub->publish(make_image(11, 11));
  ASSERT_TRUE(spin_until(adapter, helper, [&] {return observations.size() >= 1u;}));
  ASSERT_EQ(observations.size(), 1u);
  EXPECT_EQ(observations.front().source_image.header.stamp.sec, 11);
  EXPECT_EQ(observations.front().tracked_objects.front().class_label, "person");
}

TEST(TrackedObservationAdapter, PreservesTrackIdsAcrossSynchronizedUpdates)
{
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("image_topic", "/camera/image_raw");
  opts.append_parameter_override("detections_topic", "/detections");
  opts.append_parameter_override("tracked_observation_topic", "/tracked_observation");
  auto adapter = std::make_shared<edge_vlm_ros::TrackedObservationAdapter>(opts);
  auto helper = std::make_shared<rclcpp::Node>("tracked_observation_adapter_track_test_helper");

  std::vector<edge_vlm_ros::msg::TrackedObservation> observations;
  auto sub = helper->create_subscription<edge_vlm_ros::msg::TrackedObservation>(
    "/tracked_observation", rclcpp::SystemDefaultsQoS(),
    [&](edge_vlm_ros::msg::TrackedObservation::SharedPtr msg) {
      observations.push_back(*msg);
    });

  rclcpp::QoS qos{rclcpp::KeepLast(10)};
  qos.best_effort();
  auto image_pub = helper->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
  auto detections_pub = helper->create_publisher<vision_msgs::msg::Detection2DArray>("/detections", qos);

  std::this_thread::sleep_for(100ms);
  image_pub->publish(make_image(1, 11));
  detections_pub->publish(make_detections(1, 10.0, "person"));
  ASSERT_TRUE(spin_until(adapter, helper, [&] {return observations.size() >= 1u;}));

  image_pub->publish(make_image(2, 22));
  detections_pub->publish(make_detections(2, 10.5, "person"));
  ASSERT_TRUE(spin_until(adapter, helper, [&] {return observations.size() >= 2u;}));

  ASSERT_EQ(observations[0].tracked_objects.size(), 1u);
  ASSERT_EQ(observations[1].tracked_objects.size(), 1u);
  EXPECT_EQ(observations[0].tracked_objects[0].track_id, observations[1].tracked_objects[0].track_id);
  EXPECT_EQ(observations[1].source_sequence, 2u);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
