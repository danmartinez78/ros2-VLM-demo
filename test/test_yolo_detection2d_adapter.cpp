// Copyright 2025 edge_vlm_ros contributors

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <yolo_msgs/msg/detection.hpp>
#include <yolo_msgs/msg/detection_array.hpp>

#include "edge_vlm_ros/yolo_detection2d_adapter.hpp"

namespace
{

yolo_msgs::msg::DetectionArray make_input()
{
  yolo_msgs::msg::DetectionArray msg;
  msg.header.stamp = rclcpp::Time(123, 456, RCL_ROS_TIME);
  msg.header.frame_id = "camera";

  yolo_msgs::msg::Detection det;
  det.class_id = 7;
  det.class_name = "person";
  det.score = 0.85;
  det.bbox.center.position.x = 10.5;
  det.bbox.center.position.y = 20.5;
  det.bbox.size.x = 30.0;
  det.bbox.size.y = 40.0;
  msg.detections.push_back(det);
  return msg;
}

}  // namespace

TEST(YoloDetection2DAdapter, ConvertsAndPreservesTimestamp)
{
  const auto input = make_input();
  const auto output = edge_vlm_ros::convert_yolo_detection_array(input);

  EXPECT_EQ(output.header.stamp.sec, input.header.stamp.sec);
  EXPECT_EQ(output.header.stamp.nanosec, input.header.stamp.nanosec);
  EXPECT_EQ(output.header.frame_id, input.header.frame_id);
  ASSERT_EQ(output.detections.size(), 1u);
  const auto & detection = output.detections.front();
  EXPECT_DOUBLE_EQ(detection.bbox.center.position.x, 10.5);
  EXPECT_DOUBLE_EQ(detection.bbox.center.position.y, 20.5);
  EXPECT_DOUBLE_EQ(detection.bbox.size_x, 30.0);
  EXPECT_DOUBLE_EQ(detection.bbox.size_y, 40.0);
  ASSERT_EQ(detection.results.size(), 1u);
  EXPECT_EQ(detection.results.front().hypothesis.class_id, "person");
  EXPECT_DOUBLE_EQ(detection.results.front().hypothesis.score, 0.85);
}

TEST(YoloDetection2DAdapter, FallsBackToNumericClassIdWhenNameMissing)
{
  auto input = make_input();
  input.detections.front().class_name.clear();
  input.detections.front().class_id = 42;

  const auto output = edge_vlm_ros::convert_yolo_detection_array(input);

  ASSERT_EQ(output.detections.size(), 1u);
  ASSERT_EQ(output.detections.front().results.size(), 1u);
  EXPECT_EQ(output.detections.front().results.front().hypothesis.class_id, "42");
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
