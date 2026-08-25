// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

/// CPU-only static tests for TensorRT Edge-LLM backend multi-frame request
/// construction logic.  These tests do NOT require TensorRT, CUDA, or any
/// NVIDIA hardware; they validate only the request-shaping helpers exposed
/// through edge_vlm_ros::detail in tensorrt_edge_llm_backend.hpp.

#include <gtest/gtest.h>

#include "edge_vlm_ros/inference_backend.hpp"
#include "edge_vlm_ros/tensorrt_edge_llm_backend.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>

namespace
{

using edge_vlm_ros::InferenceRequest;
namespace detail = edge_vlm_ros::detail;

InferenceRequest make_request(std::size_t extra_count)
{
  InferenceRequest req;
  req.image = cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
  req.prompt = "describe the scene";
  req.max_generate_length = 32;
  for (std::size_t i = 0; i < extra_count; ++i) {
    const auto val = static_cast<uint8_t>(i + 50);
    req.extra_images.emplace_back(cv::Mat(4, 4, CV_8UC3, cv::Scalar(val, val, val)));
  }
  return req;
}

TEST(TrtBackendMultiframe, SingleFrameHasOneImageContentItem)
{
  auto req = make_request(0);
  EXPECT_EQ(detail::media_content_count(req), 1u);
}

TEST(TrtBackendMultiframe, TwoFramesHaveTwoImageContentItems)
{
  auto req = make_request(1);
  EXPECT_EQ(detail::media_content_count(req), 2u);
}

TEST(TrtBackendMultiframe, FourFramesHaveFourImageContentItems)
{
  auto req = make_request(3);
  EXPECT_EQ(detail::media_content_count(req), 4u);
}

TEST(TrtBackendMultiframe, EightFramesHaveEightImageContentItems)
{
  auto req = make_request(7);
  EXPECT_EQ(detail::media_content_count(req), 8u);
}

TEST(TrtBackendMultiframe, ImageContentCountMatchesExpectedBufferCount)
{
  for (std::size_t extra : {0u, 1u, 3u, 7u}) {
    auto req = make_request(extra);
    EXPECT_EQ(detail::media_content_count(req), extra + 1u)
      << "extra_count=" << extra;
    EXPECT_EQ(detail::image_buffer_count(req), extra + 1u)
      << "extra_count=" << extra;
  }
}

TEST(TrtBackendMultiframe, SingleFrameUserMessageHasTwoContents)
{
  auto req = make_request(0);
  EXPECT_EQ(detail::user_message_content_count(req), 2u);
}

TEST(TrtBackendMultiframe, F2UserMessageHasThreeContents)
{
  auto req = make_request(1);
  EXPECT_EQ(detail::user_message_content_count(req), 3u);
}

TEST(TrtBackendMultiframe, F4UserMessageHasFiveContents)
{
  auto req = make_request(3);
  EXPECT_EQ(detail::user_message_content_count(req), 5u);
}

TEST(TrtBackendMultiframe, F8UserMessageHasNineContents)
{
  auto req = make_request(7);
  EXPECT_EQ(detail::user_message_content_count(req), 9u);
}

TEST(TrtBackendMultiframe, ExtraImagesPreserveInsertionOrder)
{
  InferenceRequest req;
  req.image = cv::Mat(2, 2, CV_8UC3, cv::Scalar(10, 10, 10));
  req.prompt = "test";
  req.max_generate_length = 8;

  for (int i = 0; i < 3; ++i) {
    req.extra_images.emplace_back(
      cv::Mat(2, 2, CV_8UC3, cv::Scalar(20 + i * 10, 0, 0)));
  }

  ASSERT_EQ(req.extra_images.size(), 3u);
  EXPECT_EQ(req.extra_images[0].at<cv::Vec3b>(0, 0)[0], 20);
  EXPECT_EQ(req.extra_images[1].at<cv::Vec3b>(0, 0)[0], 30);
  EXPECT_EQ(req.extra_images[2].at<cv::Vec3b>(0, 0)[0], 40);
  EXPECT_EQ(detail::user_message_content_count(req), 5u);
}

TEST(TrtBackendMultiframe, ContentCountConsistency)
{
  for (std::size_t extra : {0u, 1u, 2u, 3u, 4u, 7u}) {
    auto req = make_request(extra);
    EXPECT_EQ(
      detail::media_content_count(req) + 1u,
      detail::user_message_content_count(req))
      << "extra_count=" << extra;
  }
}

TEST(TrtBackendMultiframe, TemporalSequenceUsesSingleVideoContentAndBuffer)
{
  auto req = make_request(3);
  req.sequence_type = edge_vlm_ros::TemporalSequenceType::kVideo;
  EXPECT_TRUE(detail::uses_native_video_encoding(req));
  EXPECT_EQ(detail::media_content_count(req), 1u);
  EXPECT_EQ(detail::image_buffer_count(req), 1u);
  EXPECT_STREQ(detail::media_content_type(req), "video");
  EXPECT_EQ(detail::user_message_content_count(req), 2u);
}

TEST(TrtBackendMultiframe, TemporalSequenceCarriesEffectiveNativeVideoFps)
{
  auto req = make_request(1);
  req.sequence_type = edge_vlm_ros::TemporalSequenceType::kTemporalImages;
  req.fps = 8.0;
  req.frame_timestamps_sec = {0.0, 0.125};
  ASSERT_TRUE(detail::infer_effective_video_fps(req).has_value());
  EXPECT_DOUBLE_EQ(*detail::infer_effective_video_fps(req), 8.0);
}

TEST(TrtBackendMultiframe, TemporalSequenceDerivesFpsFromTimestampsWhenNotProvided)
{
  auto req = make_request(2);
  req.sequence_type = edge_vlm_ros::TemporalSequenceType::kVideo;
  req.frame_timestamps_sec = {0.0, 0.1, 0.2};
  ASSERT_TRUE(detail::infer_effective_video_fps(req).has_value());
  EXPECT_NEAR(*detail::infer_effective_video_fps(req), 10.0, 1e-9);
}

TEST(TrtBackendMultiframe, TemporalImagesAllowConsistentFpsAndTimestamps)
{
  auto req = make_request(1);
  req.sequence_type = edge_vlm_ros::TemporalSequenceType::kTemporalImages;
  req.fps = 4.0;
  req.frame_timestamps_sec = {0.0, 0.25};
  EXPECT_NO_THROW(edge_vlm_ros::detail::validate_temporal_metadata(req));
}

TEST(TrtBackendMultiframe, ImagesRejectTemporalMetadata)
{
  auto req = make_request(0);
  req.fps = 8.0;
  EXPECT_THROW(edge_vlm_ros::detail::validate_temporal_metadata(req), std::runtime_error);
}

TEST(TrtBackendMultiframe, RejectsTimestampCountMismatch)
{
  auto req = make_request(2);
  req.sequence_type = edge_vlm_ros::TemporalSequenceType::kTemporalImages;
  req.frame_timestamps_sec = {0.0, 0.1};
  EXPECT_THROW(edge_vlm_ros::detail::validate_temporal_metadata(req), std::runtime_error);
}

TEST(TrtBackendMultiframe, RejectsFpsTimestampConflict)
{
  auto req = make_request(1);
  req.sequence_type = edge_vlm_ros::TemporalSequenceType::kTemporalImages;
  req.fps = 10.0;
  req.frame_timestamps_sec = {0.0, 0.2};
  EXPECT_THROW(edge_vlm_ros::detail::validate_temporal_metadata(req), std::runtime_error);
}

}  // namespace

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
