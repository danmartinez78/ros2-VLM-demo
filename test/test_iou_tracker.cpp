#include <gtest/gtest.h>

#include <vector>

#include "edge_vlm_ros/iou_tracker.hpp"

using edge_vlm_ros::IouTracker;
using edge_vlm_ros::TrackedBox;

TEST(IouTracker, PreservesStableTrackIdsAcrossUpdates)
{
  IouTracker tracker;
  auto first = tracker.update({TrackedBox{"person", 0.9f, 10.0f, 10.0f, 4.0f, 4.0f}});
  auto second = tracker.update({TrackedBox{"person", 0.8f, 10.5f, 10.0f, 4.0f, 4.0f}});

  ASSERT_EQ(first.size(), 1u);
  ASSERT_EQ(second.size(), 1u);
  EXPECT_EQ(first[0].track_id, second[0].track_id);
}

TEST(IouTracker, ClassAwareAssociationPreventsCrossClassMatch)
{
  IouTracker tracker;
  auto first = tracker.update({TrackedBox{"person", 0.9f, 10.0f, 10.0f, 4.0f, 4.0f}});
  auto second = tracker.update({TrackedBox{"forklift", 0.9f, 10.0f, 10.0f, 4.0f, 4.0f}});

  ASSERT_EQ(first.size(), 1u);
  ASSERT_EQ(second.size(), 2u);
  EXPECT_NE(second[0].track_id, second[1].track_id);
}

TEST(IouTracker, CoastsAndExpiresTracks)
{
  IouTracker tracker(IouTracker::Options{0.3f, 1u, true});
  auto first = tracker.update({TrackedBox{"person", 0.9f, 10.0f, 10.0f, 4.0f, 4.0f}});
  auto coasted = tracker.update({});
  auto expired = tracker.update({});

  ASSERT_EQ(first.size(), 1u);
  ASSERT_EQ(coasted.size(), 1u);
  EXPECT_EQ(coasted[0].coast_age, 1u);
  EXPECT_TRUE(expired.empty());
}
