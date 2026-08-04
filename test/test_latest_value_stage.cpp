#include <gtest/gtest.h>

#include <string>

#include "edge_vlm_ros/latest_value_stage.hpp"

TEST(LatestValueStage, ReplacesPendingValueAndTracksHighWater)
{
  edge_vlm_ros::LatestValueStage<std::string> stage;
  stage.note_activate();
  stage.push("first");
  stage.push("second");

  auto pending = stage.take_pending();
  ASSERT_TRUE(pending.has_value());
  EXPECT_EQ(*pending, "second");
  EXPECT_EQ(stage.stats().pending_superseded, 1u);
  EXPECT_EQ(stage.stats().high_water_mark, 2u);
}

TEST(LatestValueStage, ActiveSupersessionCounterIsExplicit)
{
  edge_vlm_ros::LatestValueStage<int> stage;
  stage.note_activate();
  stage.note_active_superseded();
  stage.push(42);

  EXPECT_EQ(stage.stats().active_superseded, 1u);
  EXPECT_EQ(stage.stats().accepted, 1u);
}
