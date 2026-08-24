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

// ── helper: make a minimal non-empty request with N extra images ──────────

InferenceRequest make_request(std::size_t extra_count)
{
  InferenceRequest req;
  req.image = cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
  req.prompt = "describe the scene";
  req.max_generate_length = 32;
  for (std::size_t i = 0; i < extra_count; ++i) {
    // Each extra frame has a distinct pixel value so temporal order is
    // distinguishable in future integration tests.
    const auto val = static_cast<uint8_t>(i + 50);
    req.extra_images.emplace_back(cv::Mat(4, 4, CV_8UC3, cv::Scalar(val, val, val)));
  }
  return req;
}

// ─────────────────────────────────────────────────────────────────────────────
// image_content_count
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrtBackendMultiframe, SingleFrameHasOneImageContentItem)
{
  auto req = make_request(0);
  EXPECT_EQ(detail::image_content_count(req), 1u);
}

TEST(TrtBackendMultiframe, TwoFramesHaveTwoImageContentItems)
{
  auto req = make_request(1);  // primary + 1 extra = F2
  EXPECT_EQ(detail::image_content_count(req), 2u);
}

TEST(TrtBackendMultiframe, FourFramesHaveFourImageContentItems)
{
  auto req = make_request(3);  // primary + 3 extra = F4
  EXPECT_EQ(detail::image_content_count(req), 4u);
}

TEST(TrtBackendMultiframe, EightFramesHaveEightImageContentItems)
{
  auto req = make_request(7);  // primary + 7 extra = F8
  EXPECT_EQ(detail::image_content_count(req), 8u);
}

// image_content_count == imageBuffers count (one buffer per frame)
TEST(TrtBackendMultiframe, ImageContentCountMatchesExpectedBufferCount)
{
  for (std::size_t extra : {0u, 1u, 3u, 7u}) {
    auto req = make_request(extra);
    EXPECT_EQ(detail::image_content_count(req), extra + 1u)
      << "extra_count=" << extra;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// user_message_content_count  (image items + 1 text item)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrtBackendMultiframe, SingleFrameUserMessageHasTwoContents)
{
  // 1 image item + 1 text item
  auto req = make_request(0);
  EXPECT_EQ(detail::user_message_content_count(req), 2u);
}

TEST(TrtBackendMultiframe, F2UserMessageHasThreeContents)
{
  // 2 image items + 1 text item
  auto req = make_request(1);
  EXPECT_EQ(detail::user_message_content_count(req), 3u);
}

TEST(TrtBackendMultiframe, F4UserMessageHasFiveContents)
{
  // 4 image items + 1 text item
  auto req = make_request(3);
  EXPECT_EQ(detail::user_message_content_count(req), 5u);
}

TEST(TrtBackendMultiframe, F8UserMessageHasNineContents)
{
  // 8 image items + 1 text item
  auto req = make_request(7);
  EXPECT_EQ(detail::user_message_content_count(req), 9u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Temporal order: extra_images[0] is the second frame (index 1)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrtBackendMultiframe, ExtraImagesPreserveInsertionOrder)
{
  InferenceRequest req;
  req.image = cv::Mat(2, 2, CV_8UC3, cv::Scalar(10, 10, 10));
  req.prompt = "test";
  req.max_generate_length = 8;

  // Push frames with distinct identifiable values.
  for (int i = 0; i < 3; ++i) {
    req.extra_images.emplace_back(
      cv::Mat(2, 2, CV_8UC3, cv::Scalar(20 + i * 10, 0, 0)));
  }

  // Verify sizes and order preservation via the extra_images vector itself.
  ASSERT_EQ(req.extra_images.size(), 3u);
  EXPECT_EQ(req.extra_images[0].at<cv::Vec3b>(0, 0)[0], 20);
  EXPECT_EQ(req.extra_images[1].at<cv::Vec3b>(0, 0)[0], 30);
  EXPECT_EQ(req.extra_images[2].at<cv::Vec3b>(0, 0)[0], 40);

  // Expected content count: 4 image + 1 text = 5
  EXPECT_EQ(detail::user_message_content_count(req), 5u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Consistency: image_content_count + 1 == user_message_content_count
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrtBackendMultiframe, ContentCountConsistency)
{
  for (std::size_t extra : {0u, 1u, 2u, 3u, 4u, 7u}) {
    auto req = make_request(extra);
    EXPECT_EQ(
      detail::image_content_count(req) + 1u,
      detail::user_message_content_count(req))
      << "extra_count=" << extra;
  }
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
