// Copyright 2025 edge_vlm_ros contributors

#include <gtest/gtest.h>

#include "edge_vlm_ros/inference_watchdog.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

namespace
{
using edge_vlm_ros::InferenceWatchdog;

// Helper: build a custom expire handler that records the callback arguments
// and signals a future without terminating the process.
auto make_recording_handler(
  std::promise<void> & fired_promise,
  int & out_deadline,
  uint64_t & out_request_id)
{
  return [&fired_promise, &out_deadline, &out_request_id](int deadline, uint64_t request_id) {
           out_deadline = deadline;
           out_request_id = request_id;
           try {
             fired_promise.set_value();
           } catch (...) {}
         };
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// InferenceWatchdog unit tests
//
// These tests exercise the actual watchdog implementation without calling
// std::_Exit.  A custom expire handler records the callback parameters and
// signals a std::promise so the test can assert on the result.
// ─────────────────────────────────────────────────────────────────────────────

TEST(InferenceWatchdog, DoesNotFireWhenCancelledBeforeDeadline)
{
  std::atomic<bool> expired{false};

  {
    InferenceWatchdog watchdog(
      5,   // 5-second deadline — much longer than the cancel delay
      42,
      [&expired](int, uint64_t) {expired.store(true);});

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    watchdog.cancel();   // fires before the deadline
    // Destructor joins the watchdog thread
  }

  EXPECT_FALSE(expired.load()) << "Watchdog must not fire when cancelled before the deadline";
}

TEST(InferenceWatchdog, FiresWhenDeadlineExpires)
{
  std::promise<void> fired_promise;
  auto fired_future = fired_promise.get_future();
  int fired_deadline = 0;
  uint64_t fired_request_id = 0;

  {
    InferenceWatchdog watchdog(
      1,   // 1-second deadline
      99,
      make_recording_handler(fired_promise, fired_deadline, fired_request_id));

    // Do NOT cancel — let the deadline expire.
    ASSERT_EQ(
      fired_future.wait_for(std::chrono::seconds(4)),
      std::future_status::ready)
      << "Watchdog must fire within 4 s when a 1-second deadline expires";
  }

  EXPECT_EQ(fired_deadline, 1) << "on_expire must receive the configured deadline";
  EXPECT_EQ(fired_request_id, 99U) << "on_expire must receive the correct request_id";
}

TEST(InferenceWatchdog, FiresWithCorrectRequestId)
{
  std::promise<void> fired_promise;
  auto fired_future = fired_promise.get_future();
  int fired_deadline = 0;
  uint64_t fired_request_id = 0;

  {
    InferenceWatchdog watchdog(
      1,
      0xDEADBEEFCAFE0042ULL,
      make_recording_handler(fired_promise, fired_deadline, fired_request_id));

    ASSERT_EQ(
      fired_future.wait_for(std::chrono::seconds(4)),
      std::future_status::ready);
  }

  EXPECT_EQ(fired_request_id, 0xDEADBEEFCAFE0042ULL);
}

TEST(InferenceWatchdog, DestructorCancelsAndJoins)
{
  // Watchdog with a very long deadline — destroying it immediately must not
  // block indefinitely or call the expire handler.
  std::atomic<bool> expired{false};

  {
    auto watchdog = std::make_unique<InferenceWatchdog>(
      120,   // 120-second deadline — will never fire naturally in a test
      1,
      [&expired](int, uint64_t) {expired.store(true);});

    watchdog.reset();  // destroy immediately
  }

  EXPECT_FALSE(expired.load()) << "Destructor must cancel the watchdog, not fire it";
}

TEST(InferenceWatchdog, MultipleExplicitCancelCallsAreIdempotent)
{
  std::atomic<bool> expired{false};

  {
    InferenceWatchdog watchdog(
      5,
      1,
      [&expired](int, uint64_t) {expired.store(true);});

    watchdog.cancel();
    watchdog.cancel();  // second call must be a silent no-op
    watchdog.cancel();
  }

  EXPECT_FALSE(expired.load());
}

TEST(InferenceWatchdog, CancelAfterDeadlineIsIdempotent)
{
  // Let the watchdog fire, then explicitly cancel afterwards — must not throw.
  std::promise<void> fired_promise;
  auto fired_future = fired_promise.get_future();
  int fired_deadline = 0;
  uint64_t fired_request_id = 0;

  InferenceWatchdog watchdog(
    1,
    7,
    make_recording_handler(fired_promise, fired_deadline, fired_request_id));

  ASSERT_EQ(
    fired_future.wait_for(std::chrono::seconds(4)),
    std::future_status::ready);

  EXPECT_NO_THROW(watchdog.cancel())
    << "cancel() after expiry must be a no-op and must not throw";
}
