// Copyright 2025 edge_vlm_ros contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace edge_vlm_ros
{

struct TrackedBox
{
  std::string class_label;
  float confidence{0.0f};
  float center_x{0.0f};
  float center_y{0.0f};
  float width{0.0f};
  float height{0.0f};
};

struct TrackState
{
  uint64_t track_id{0};
  TrackedBox box;
  float velocity_x{0.0f};
  float velocity_y{0.0f};
  uint32_t age{0};
  uint32_t coast_age{0};
};

class IouTracker
{
public:
  struct Options
  {
    float min_iou{0.3f};
    uint32_t max_coast_age{1};
    bool class_aware{true};
  };

  IouTracker() = default;
  explicit IouTracker(Options options)
  : options_(options) {}

  std::vector<TrackState> update(const std::vector<TrackedBox> & detections)
  {
    std::vector<bool> matched_detection(detections.size(), false);
    std::vector<TrackState> next_tracks;
    next_tracks.reserve(std::max(tracks_.size(), detections.size()));

    for (const auto & track : tracks_) {
      const float predicted_x = track.box.center_x + track.velocity_x;
      const float predicted_y = track.box.center_y + track.velocity_y;

      size_t best_index = detections.size();
      float best_iou = -1.0f;
      for (size_t i = 0; i < detections.size(); ++i) {
        if (matched_detection[i]) {
          continue;
        }
        if (options_.class_aware && detections[i].class_label != track.box.class_label) {
          continue;
        }
        const float score = iou(predicted_x, predicted_y, track.box.width, track.box.height, detections[i]);
        if (score > best_iou) {
          best_iou = score;
          best_index = i;
        }
      }

      if (best_index < detections.size() && best_iou >= options_.min_iou) {
        matched_detection[best_index] = true;
        TrackState updated = track;
        updated.velocity_x = detections[best_index].center_x - track.box.center_x;
        updated.velocity_y = detections[best_index].center_y - track.box.center_y;
        updated.box = detections[best_index];
        updated.age += 1;
        updated.coast_age = 0;
        next_tracks.push_back(updated);
      } else if (track.coast_age < options_.max_coast_age) {
        TrackState coasted = track;
        coasted.box.center_x = predicted_x;
        coasted.box.center_y = predicted_y;
        coasted.age += 1;
        coasted.coast_age += 1;
        next_tracks.push_back(coasted);
      }
    }

    for (size_t i = 0; i < detections.size(); ++i) {
      if (matched_detection[i]) {
        continue;
      }
      TrackState created;
      created.track_id = next_track_id_++;
      created.box = detections[i];
      created.age = 1;
      next_tracks.push_back(created);
    }

    tracks_ = next_tracks;
    return tracks_;
  }

private:
  static float iou(
    float center_x,
    float center_y,
    float width,
    float height,
    const TrackedBox & detection)
  {
    const float left_a = center_x - width / 2.0f;
    const float right_a = center_x + width / 2.0f;
    const float top_a = center_y - height / 2.0f;
    const float bottom_a = center_y + height / 2.0f;

    const float left_b = detection.center_x - detection.width / 2.0f;
    const float right_b = detection.center_x + detection.width / 2.0f;
    const float top_b = detection.center_y - detection.height / 2.0f;
    const float bottom_b = detection.center_y + detection.height / 2.0f;

    const float intersection_w = std::max(0.0f, std::min(right_a, right_b) - std::max(left_a, left_b));
    const float intersection_h = std::max(0.0f, std::min(bottom_a, bottom_b) - std::max(top_a, top_b));
    const float intersection = intersection_w * intersection_h;
    const float area_a = std::max(0.0f, width) * std::max(0.0f, height);
    const float area_b = std::max(0.0f, detection.width) * std::max(0.0f, detection.height);
    const float denom = area_a + area_b - intersection;
    if (denom <= std::numeric_limits<float>::epsilon()) {
      return 0.0f;
    }
    return intersection / denom;
  }

  Options options_;
  uint64_t next_track_id_{1};
  std::vector<TrackState> tracks_;
};

}  // namespace edge_vlm_ros
