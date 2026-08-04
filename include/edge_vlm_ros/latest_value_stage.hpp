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

#pragma once

#include <cstdint>
#include <optional>
#include <utility>

namespace edge_vlm_ros
{

template<typename T>
class LatestValueStage
{
public:
  struct Stats
  {
    uint64_t accepted{0};
    uint64_t active_superseded{0};
    uint64_t pending_superseded{0};
    uint64_t high_water_mark{0};
  };

  void note_activate()
  {
    active_ = true;
  }

  void clear_active()
  {
    active_ = false;
  }

  void push(T value)
  {
    ++stats_.accepted;
    uint64_t occupancy = active_ ? 1u : 0u;
    if (pending_.has_value()) {
      ++stats_.pending_superseded;
    }
    pending_ = std::move(value);
    ++occupancy;
    if (occupancy > stats_.high_water_mark) {
      stats_.high_water_mark = occupancy;
    }
  }

  std::optional<T> take_pending()
  {
    if (!pending_.has_value()) {
      return std::nullopt;
    }
    active_ = true;
    auto value = std::move(pending_);
    pending_.reset();
    return value;
  }

  void note_active_superseded()
  {
    ++stats_.active_superseded;
  }

  bool has_pending() const
  {
    return pending_.has_value();
  }

  const Stats & stats() const
  {
    return stats_;
  }

private:
  bool active_{false};
  std::optional<T> pending_;
  Stats stats_;
};

}  // namespace edge_vlm_ros
