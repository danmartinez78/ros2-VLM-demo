// Copyright 2025 edge_vlm_ros contributors
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <thread>

namespace edge_vlm_ros
{

/// Worker-side inference deadline watchdog.
///
/// Starts a background thread that waits for cancel() to be called within
/// @p deadline_seconds. If the deadline expires before cancel() is called,
/// @p on_expire is invoked on the watchdog thread.
///
/// **Production usage**: pass watchdog_exit_on_expire as @p on_expire; it emits
/// a structured diagnostic and calls std::_Exit(1) to bypass all C++ destructors
/// and atexit handlers, avoiding a hang on wedged CUDA state.
///
/// **Test usage**: pass a custom @p on_expire that records the expiry without
/// terminating the process.
class InferenceWatchdog
{
public:
  using ExpireHandler = std::function<void(int deadline_seconds, uint64_t request_id)>;

  /// Starts the watchdog thread immediately.
  InferenceWatchdog(int deadline_seconds, uint64_t request_id, ExpireHandler on_expire)
  : deadline_seconds_(deadline_seconds),
    request_id_(request_id),
    on_expire_(std::move(on_expire))
  {
    thread_ = std::thread([this]() {run();});
  }

  /// Cancels the watchdog and joins the background thread.
  /// Blocks until the thread has exited.
  ~InferenceWatchdog()
  {
    cancel();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  InferenceWatchdog(const InferenceWatchdog &) = delete;
  InferenceWatchdog & operator=(const InferenceWatchdog &) = delete;

  /// Signals that inference completed within the deadline.
  /// Safe to call multiple times; subsequent calls are silent no-ops.
  void cancel()
  {
    try {
      done_.set_value();
    } catch (const std::future_error &) {
      // Already satisfied — ignore.
    }
  }

private:
  void run()
  {
    if (done_.get_future().wait_for(std::chrono::seconds(deadline_seconds_)) ==
      std::future_status::timeout)
    {
      on_expire_(deadline_seconds_, request_id_);
    }
  }

  int deadline_seconds_;
  uint64_t request_id_;
  ExpireHandler on_expire_;
  std::promise<void> done_;
  std::thread thread_;
};

/// Default production expire handler.
///
/// Emits a structured diagnostic to stderr then calls std::_Exit(1).
///
/// std::_Exit is preferred over std::quick_exit because:
/// - std::_Exit bypasses ALL cleanup (C++ destructors, std::atexit handlers,
///   and std::at_quick_exit handlers). This guarantees that no registered
///   cleanup routine can attempt to teardown wedged CUDA state and block.
/// - std::quick_exit still invokes functions registered with at_quick_exit().
///   If a third-party library (e.g., CUDA runtime) registers an at_quick_exit
///   handler that touches the GPU, quick_exit would hang on the same wedged
///   state we are trying to escape.
/// - TensorRT Edge-LLM's handleRequest() exposes no supported cancellation
///   API; process-level isolation via _Exit is the only safe mechanism.
/// - The OS reclaims all file descriptors. The socket file is removed by the
///   replacement worker via ::unlink() at startup.
inline void watchdog_exit_on_expire(int deadline_seconds, uint64_t request_id)
{
  std::cerr
    << "[edge_vlm_server] WATCHDOG: inference deadline ("
    << deadline_seconds << "s) expired"
    << " request_id=" << request_id
    << "; self-terminating for clean respawn\n";
  std::cerr.flush();
  std::_Exit(1);
}

}  // namespace edge_vlm_ros
