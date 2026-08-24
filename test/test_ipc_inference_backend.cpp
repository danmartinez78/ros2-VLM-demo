// Copyright 2025 edge_vlm_ros contributors

#include <gtest/gtest.h>

#include "edge_vlm_ros/ipc_inference_backend.hpp"
#include "edge_vlm_ros/ipc_protocol.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>
#include <unistd.h>

namespace
{
using edge_vlm_ros::HistoryEntry;
using edge_vlm_ros::InferenceRequest;
using edge_vlm_ros::IpcInferenceBackend;
using edge_vlm_ros::IpcInferenceConfig;
namespace ipc = edge_vlm_ros::ipc;

std::atomic<uint64_t> g_path_counter{0};

std::string make_socket_path()
{
  const uint64_t id = g_path_counter.fetch_add(1, std::memory_order_relaxed);
  return "/tmp/edge-vlm-ipc-test-" + std::to_string(static_cast<long long>(::getpid())) + "-" +
         std::to_string(static_cast<long long>(id)) + ".sock";
}

InferenceRequest make_request(std::string prompt = "describe")
{
  InferenceRequest request;
  request.image = cv::Mat(4, 5, CV_8UC3, cv::Scalar(10, 20, 30));
  request.prompt = std::move(prompt);
  request.max_generate_length = 32;
  request.temperature = 0.4F;
  request.top_p = 0.9F;
  request.top_k = 40;
  return request;
}

class OneClientWorker
{
public:
  using SessionHandler = std::function<void(int)>;

  explicit OneClientWorker(SessionHandler handler, std::string path = make_socket_path())
  : path_(std::move(path)), handler_(std::move(handler))
  {
    thread_ = std::thread([this]() {run();});
    if (ready_future_.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      request_stop();
      join_thread();
      throw std::runtime_error("failed to start test worker");
    }
    if (!ready_future_.get()) {
      request_stop();
      join_thread();
      throw std::runtime_error("failed to start test worker");
    }
  }

  ~OneClientWorker()
  {
    request_stop();
    join_thread();
    ::unlink(path_.c_str());
  }

  std::string const & socket_path() const {return path_;}

  void join_and_rethrow()
  {
    request_stop();
    join_thread();
    if (worker_exception_) {
      std::rethrow_exception(worker_exception_);
    }
  }

private:
  void request_stop() noexcept
  {
    stop_requested_.store(true, std::memory_order_relaxed);
    const int fd = server_fd_.exchange(-1, std::memory_order_relaxed);
    if (fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
  }

  void join_thread()
  {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void run()
  {
    bool ready_reported = false;
    try {
      ::unlink(path_.c_str());
      int server_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
      if (server_fd < 0) {
        ready_promise_.set_value(false);
        return;
      }
      server_fd_.store(server_fd, std::memory_order_relaxed);

      sockaddr_un address{};
      address.sun_family = AF_UNIX;
      std::strncpy(address.sun_path, path_.c_str(), sizeof(address.sun_path) - 1);

      if (::bind(server_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        ::listen(server_fd, 1) != 0)
      {
        ready_promise_.set_value(false);
        request_stop();
        return;
      }

      ready_promise_.set_value(true);
      ready_reported = true;

      while (!stop_requested_.load(std::memory_order_relaxed)) {
        pollfd waiter{};
        waiter.fd = server_fd;
        waiter.events = POLLIN;
        const int poll_result = ::poll(&waiter, 1, 100);
        if (poll_result == 0) {
          continue;
        }
        if (poll_result < 0) {
          if (errno == EINTR) {
            continue;
          }
          throw std::runtime_error("poll failed in test worker");
        }
        if ((waiter.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
          break;
        }
        int client_fd = ::accept4(server_fd, nullptr, nullptr, SOCK_CLOEXEC);
        if (client_fd < 0) {
          if (errno == EINTR) {
            continue;
          }
          if (stop_requested_.load(std::memory_order_relaxed)) {
            break;
          }
          throw std::runtime_error("accept failed in test worker");
        }
        handler_(client_fd);
        ::close(client_fd);
        break;
      }
      request_stop();
    } catch (...) {
      if (!ready_reported) {
        try {
          ready_promise_.set_value(false);
        } catch (...) {
        }
      }
      worker_exception_ = std::current_exception();
      request_stop();
    }
  }

  std::string path_;
  SessionHandler handler_;
  std::thread thread_;
  std::promise<bool> ready_promise_;
  std::future<bool> ready_future_{ready_promise_.get_future()};
  std::atomic<int> server_fd_{-1};
  std::atomic<bool> stop_requested_{false};
  std::exception_ptr worker_exception_;
};

void read_request_frame(int fd, ipc::RequestHeader & header, std::vector<uint8_t> & image, std::string & prompt)
{
  ipc::read_all(fd, &header, sizeof(header));
  image.resize(header.image_bytes);
  prompt.resize(header.prompt_bytes);
  ipc::read_all(fd, image.data(), image.size());
  ipc::read_all(fd, prompt.data(), prompt.size());
}

/// Read a full v2 structured request from the test worker side.
/// Inline mode (schema_flags == kSchemaFlagInline): reads [image][prompt] only.
/// Structured mode: reads [image][system_message][prompt][history entries].
struct ParsedRequest
{
  ipc::RequestHeader header;
  std::vector<uint8_t> image;
  std::vector<double> frame_timestamps_sec;
  std::string system_message;
  std::string prompt;
  std::vector<std::pair<std::string, std::string>> history;  // {user_text, asst_text}
};

ParsedRequest read_full_request(int fd)
{
  ParsedRequest parsed;
  ipc::read_all(fd, &parsed.header, sizeof(parsed.header));
  parsed.image.resize(parsed.header.image_bytes);
  ipc::read_all(fd, parsed.image.data(), parsed.image.size());
  if ((parsed.header.schema_flags & ipc::kSchemaFlagHasFrameTimestamps) != 0U) {
    parsed.frame_timestamps_sec.resize(parsed.header.timestamp_count);
    ipc::read_all(
      fd, parsed.frame_timestamps_sec.data(),
      parsed.frame_timestamps_sec.size() * sizeof(double));
  }
  if (parsed.header.system_bytes > 0) {
    parsed.system_message.resize(parsed.header.system_bytes);
    ipc::read_all(fd, parsed.system_message.data(), parsed.system_message.size());
  }
  parsed.prompt.resize(parsed.header.prompt_bytes);
  ipc::read_all(fd, parsed.prompt.data(), parsed.prompt.size());
  for (uint32_t i = 0; i < parsed.header.history_count; ++i) {
    ipc::HistoryEntryHeader entry_hdr{};
    ipc::read_all(fd, &entry_hdr, sizeof(entry_hdr));
    std::string user_text(entry_hdr.user_bytes, '\0');
    std::string asst_text(entry_hdr.asst_bytes, '\0');
    ipc::read_all(fd, user_text.data(), user_text.size());
    ipc::read_all(fd, asst_text.data(), asst_text.size());
    parsed.history.emplace_back(std::move(user_text), std::move(asst_text));
  }
  return parsed;
}

void send_success_response(int fd, uint64_t request_id, std::string text)
{
  ipc::ResponseHeader response;
  response.request_id = request_id;
  response.success = 1;
  response.text_bytes = static_cast<uint32_t>(text.size());
  response.error_bytes = 0;
  response.inference_seconds = 0.123;
  ipc::write_all(fd, &response, sizeof(response));
  ipc::write_all(fd, text.data(), text.size());
}
}  // namespace

TEST(IpcInferenceBackend, SendsExpectedRequestHeader)
{
  ipc::RequestHeader seen_header{};
  std::string seen_prompt;
  std::vector<uint8_t> seen_image;

  OneClientWorker worker([&](int client_fd) {
    read_request_frame(client_fd, seen_header, seen_image, seen_prompt);
    send_success_response(client_fd, seen_header.request_id, "ok");
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest request = make_request("inspect scene");
  ASSERT_TRUE(request.image.isContinuous());
  auto response = backend.infer(request);

  ASSERT_TRUE(response.success);
  EXPECT_EQ(response.text, "ok");
  EXPECT_EQ(seen_header.magic, ipc::kMagic);
  EXPECT_EQ(seen_header.version, ipc::kVersion);
  EXPECT_EQ(seen_header.encoding, ipc::kEncodingBgr8);
  EXPECT_EQ(seen_header.width, static_cast<uint32_t>(request.image.cols));
  EXPECT_EQ(seen_header.height, static_cast<uint32_t>(request.image.rows));
  EXPECT_EQ(seen_header.step, static_cast<uint32_t>(request.image.cols * 3));
  EXPECT_EQ(seen_header.image_bytes, request.image.total() * request.image.elemSize());
  EXPECT_EQ(seen_prompt, request.prompt);
  std::vector<uint8_t> expected_image(
    request.image.data, request.image.data + (request.image.total() * request.image.elemSize()));
  EXPECT_EQ(seen_image, expected_image);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, RejectsOversizedPromptAndImage)
{
  OneClientWorker worker([](int client_fd) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ::close(client_fd);
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 1;
  config.max_text_bytes = 32;
  config.max_image_bytes = 32;

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest oversized_prompt = make_request();
  oversized_prompt.prompt.assign(config.max_text_bytes + 1U, 'x');
  EXPECT_THROW(backend.infer(oversized_prompt), std::runtime_error);

  InferenceRequest oversized_image = make_request();
  EXPECT_THROW(backend.infer(oversized_image), std::runtime_error);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, RejectsInvalidResponseHeaders)
{
  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);

    ipc::ResponseHeader response;
    response.magic = 0xDEADBEEF;
    response.request_id = request_header.request_id;
    response.success = 1;
    response.text_bytes = 0;
    response.error_bytes = 0;
    ipc::write_all(client_fd, &response, sizeof(response));
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  EXPECT_THROW(backend.infer(make_request()), std::runtime_error);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, RejectsVersionMismatchResponseHeaders)
{
  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);

    ipc::ResponseHeader response;
    response.version = ipc::kVersion + 1U;
    response.request_id = request_header.request_id;
    response.success = 1;
    response.text_bytes = 0;
    response.error_bytes = 0;
    ipc::write_all(client_fd, &response, sizeof(response));
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  EXPECT_THROW(backend.infer(make_request()), std::runtime_error);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, RejectsMismatchedRequestIdResponseHeaders)
{
  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);

    ipc::ResponseHeader response;
    response.request_id = request_header.request_id + 1U;
    response.success = 1;
    response.text_bytes = 0;
    response.error_bytes = 0;
    ipc::write_all(client_fd, &response, sizeof(response));
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  EXPECT_THROW(backend.infer(make_request()), std::runtime_error);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, RejectsOversizedResponseFields)
{
  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);

    ipc::ResponseHeader response;
    response.request_id = request_header.request_id;
    response.success = 1;
    response.text_bytes = ipc::kMaxTextBytes + 1U;
    response.error_bytes = 0;
    ipc::write_all(client_fd, &response, sizeof(response));
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  EXPECT_THROW(backend.infer(make_request()), std::runtime_error);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, ReturnsWorkerErrorResponse)
{
  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);

    const std::string error = "worker failed";
    ipc::ResponseHeader response;
    response.request_id = request_header.request_id;
    response.success = 0;
    response.text_bytes = 0;
    response.error_bytes = static_cast<uint32_t>(error.size());
    ipc::write_all(client_fd, &response, sizeof(response));
    ipc::write_all(client_fd, error.data(), error.size());
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  auto response = backend.infer(make_request());
  EXPECT_FALSE(response.success);
  EXPECT_TRUE(response.text.empty());
  EXPECT_EQ(response.error, "worker failed");
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, ParsesTemporalEncodingFieldsFromWorkerResponse)
{
  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);

    const std::string encoding = "native_qwen3vl_video_imagedata_mrope_timestamps";
    ipc::ResponseHeader response;
    response.request_id = request_header.request_id;
    response.success = 1;
    response.text_bytes = 2;
    response.error_bytes = 0;
    response.temporal_encoding_bytes = static_cast<uint32_t>(encoding.size());
    response.temporal_fallback_used = 0U;
    ipc::write_all(client_fd, &response, sizeof(response));
    const std::string text = "ok";
    ipc::write_all(client_fd, text.data(), text.size());
    ipc::write_all(client_fd, encoding.data(), encoding.size());
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  auto response = backend.infer(make_request());
  EXPECT_TRUE(response.success);
  EXPECT_EQ(response.text, "ok");
  EXPECT_EQ(
    response.runtime_temporal_encoding,
    "native_qwen3vl_video_imagedata_mrope_timestamps");
  EXPECT_FALSE(response.temporal_fallback_used);
  EXPECT_EQ(response.requested_sequence_type, "images");
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, FailsCleanlyOnTruncatedResponsePayload)
{
  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);

    ipc::ResponseHeader response;
    response.request_id = request_header.request_id;
    response.success = 1;
    response.text_bytes = 6;
    response.error_bytes = 0;
    ipc::write_all(client_fd, &response, sizeof(response));
    const std::string partial = "abc";
    ipc::write_all(client_fd, partial.data(), partial.size());
    ::shutdown(client_fd, SHUT_WR);
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  EXPECT_THROW(backend.infer(make_request()), std::runtime_error);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, TimesOutWhenWorkerDoesNotRespond)
{
  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);
    std::this_thread::sleep_for(std::chrono::seconds(2));
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 1;

  IpcInferenceBackend backend(config);
  backend.initialize();

  auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(backend.infer(make_request()), std::runtime_error);
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_GT(elapsed, std::chrono::milliseconds(700));
  EXPECT_LT(elapsed, std::chrono::milliseconds(2200));
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, ReconnectsAfterWorkerRestart)
{
  std::string socket_path = make_socket_path();
  IpcInferenceConfig config;
  config.socket_path = socket_path;
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;
  auto backend = std::make_unique<IpcInferenceBackend>(config);
  {
    OneClientWorker worker([&](int client_fd) {
      ipc::RequestHeader request_header;
      std::vector<uint8_t> image;
      std::string prompt;
      read_request_frame(client_fd, request_header, image, prompt);
      send_success_response(client_fd, request_header.request_id, "first");
      ::shutdown(client_fd, SHUT_RDWR);
    }, socket_path);
    backend->initialize();

    auto first = backend->infer(make_request("first request"));
    ASSERT_TRUE(first.success);
    EXPECT_EQ(first.text, "first");

    EXPECT_THROW(backend->infer(make_request("expected disconnect")), std::runtime_error);
    worker.join_and_rethrow();
  }

  OneClientWorker restarted_worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);
    send_success_response(client_fd, request_header.request_id, "second");
  }, socket_path);

  auto second = backend->infer(make_request("after restart"));
  EXPECT_TRUE(second.success);
  EXPECT_EQ(second.text, "second");
  restarted_worker.join_and_rethrow();
}

// Simulates the watchdog recovery path matching production semantics:
//   1. Worker reads the request, hangs for 1 s (the "worker-side deadline"),
//      then closes the connection and exits — simulating InferenceWatchdog
//      expiry and the resulting std::_Exit(1).
//   2. The client sees an EOF (not a SO_RCVTIMEO timeout) because the worker
//      exits within the 3-second client timeout.  Exactly one error is
//      reported; the request is NOT replayed.
//   3. A replacement worker becomes available on the same socket path.
//   4. The client reconnects and succeeds without restarting edge_vlm_ros_node.
//
// Worker-side deadline (1 s) < client timeout (3 s) matches the production
// relationship: worker_inference_deadline_seconds < worker_request_timeout_seconds.
TEST(IpcInferenceBackend, TimesOutThenReconnectsAfterWorkerSelfTermination)
{
  std::string socket_path = make_socket_path();

  IpcInferenceConfig config;
  config.socket_path = socket_path;
  config.connect_timeout_seconds = 3;
  config.request_timeout_seconds = 3;   // client timeout > worker-side deadline

  auto backend = std::make_unique<IpcInferenceBackend>(config);

  // Worker 1: reads the request, waits 1 s (simulating the watchdog deadline
  // expiring), then closes the connection and exits.  The client timeout (3 s)
  // is intentionally longer so the client sees an EOF rather than a
  // SO_RCVTIMEO error — matching production semantics.
  {
    OneClientWorker wedged_worker([](int client_fd) {
      ipc::RequestHeader request_header;
      std::vector<uint8_t> image;
      std::string prompt;
      read_request_frame(client_fd, request_header, image, prompt);
      // Simulate the watchdog firing after its 1-second deadline: worker
      // stops responding and closes the socket, giving the client a clean EOF.
      std::this_thread::sleep_for(std::chrono::seconds(1));
      // Worker exits here (OneClientWorker closes client_fd on return).
    }, socket_path);

    backend->initialize();

    auto start = std::chrono::steady_clock::now();
    // Client must receive exactly one error (EOF from the exited worker).
    // The request is NOT replayed.
    EXPECT_THROW(backend->infer(make_request("wedged")), std::runtime_error);
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Must complete after the 1-second worker hang but well before the 3-second
    // client timeout.
    EXPECT_GT(elapsed, std::chrono::milliseconds(700));
    EXPECT_LT(elapsed, std::chrono::milliseconds(2500));

    wedged_worker.join_and_rethrow();
  }
  // Worker 1 has exited (socket unlinked by OneClientWorker destructor).

  // Worker 2: replacement worker available immediately on the same path.
  OneClientWorker replacement_worker([](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);
    send_success_response(client_fd, request_header.request_id, "recovered");
  }, socket_path);

  // Client must reconnect automatically and succeed without restarting
  // edge_vlm_ros_node — exactly one successful result for the next request.
  auto recovered = backend->infer(make_request("after recovery"));
  EXPECT_TRUE(recovered.success);
  EXPECT_EQ(recovered.text, "recovered");
  replacement_worker.join_and_rethrow();
}

// ── Structured-message round-trip tests (IPC schema v3) ──────────────────────

TEST(IpcInferenceBackend, InlineModeEmitsSchemaFlagsZero)
{
  ParsedRequest seen;

  OneClientWorker worker([&](int client_fd) {
    seen = read_full_request(client_fd);
    send_success_response(client_fd, seen.header.request_id, "ok");
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  // make_request() has no system_message and no history → inline mode.
  auto response = backend.infer(make_request("hello"));
  ASSERT_TRUE(response.success);

  EXPECT_EQ(seen.header.version, ipc::kVersion);
  EXPECT_EQ(seen.header.schema_flags, ipc::kSchemaFlagInline);
  EXPECT_EQ(seen.header.system_bytes, 0U);
  EXPECT_EQ(seen.header.history_count, 0U);
  EXPECT_EQ(seen.prompt, "hello");
  EXPECT_TRUE(seen.system_message.empty());
  EXPECT_TRUE(seen.history.empty());
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, StructuredModeSystemMessageRoundTrip)
{
  ParsedRequest seen;

  OneClientWorker worker([&](int client_fd) {
    seen = read_full_request(client_fd);
    send_success_response(client_fd, seen.header.request_id, "ok");
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest request = make_request("describe scene");
  request.system_message = "You are a robot vision assistant.";

  auto response = backend.infer(request);
  ASSERT_TRUE(response.success);

  EXPECT_EQ(seen.header.version, ipc::kVersion);
  EXPECT_TRUE(seen.header.schema_flags & ipc::kSchemaFlagStructured);
  EXPECT_EQ(seen.header.system_bytes, static_cast<uint32_t>(request.system_message.size()));
  EXPECT_EQ(seen.header.history_count, 0U);
  EXPECT_EQ(seen.system_message, request.system_message);
  EXPECT_EQ(seen.prompt, "describe scene");
  EXPECT_TRUE(seen.history.empty());
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, StructuredModeHistoryRoundTrip)
{
  ParsedRequest seen;

  OneClientWorker worker([&](int client_fd) {
    seen = read_full_request(client_fd);
    send_success_response(client_fd, seen.header.request_id, "ok");
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest request = make_request("current frame");
  request.system_message = "Be precise.";
  request.history.push_back(HistoryEntry{"first user turn", "first assistant reply"});
  request.history.push_back(HistoryEntry{"second user turn", "second assistant reply"});

  auto response = backend.infer(request);
  ASSERT_TRUE(response.success);

  EXPECT_EQ(seen.header.version, ipc::kVersion);
  EXPECT_TRUE(seen.header.schema_flags & ipc::kSchemaFlagStructured);
  EXPECT_EQ(seen.header.system_bytes, static_cast<uint32_t>(request.system_message.size()));
  EXPECT_EQ(seen.header.history_count, 2U);
  EXPECT_EQ(seen.system_message, "Be precise.");
  EXPECT_EQ(seen.prompt, "current frame");

  ASSERT_EQ(seen.history.size(), 2U);
  EXPECT_EQ(seen.history[0].first, "first user turn");
  EXPECT_EQ(seen.history[0].second, "first assistant reply");
  EXPECT_EQ(seen.history[1].first, "second user turn");
  EXPECT_EQ(seen.history[1].second, "second assistant reply");
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, SysCacheFlagSetOnlyWhenRequested)
{
  ParsedRequest seen_cached, seen_uncached;

  // First request: cache flag requested.
  {
    OneClientWorker worker([&](int client_fd) {
      seen_cached = read_full_request(client_fd);
      send_success_response(client_fd, seen_cached.header.request_id, "ok");
    });

    IpcInferenceConfig config;
    config.socket_path = worker.socket_path();
    config.connect_timeout_seconds = 2;
    config.request_timeout_seconds = 2;

    IpcInferenceBackend backend(config);
    backend.initialize();

    InferenceRequest request = make_request("inspect");
    request.system_message = "Sys prompt.";
    request.use_system_prompt_cache = true;

    ASSERT_TRUE(backend.infer(request).success);
    worker.join_and_rethrow();
  }

  // Second request: cache flag NOT requested.
  {
    OneClientWorker worker([&](int client_fd) {
      seen_uncached = read_full_request(client_fd);
      send_success_response(client_fd, seen_uncached.header.request_id, "ok");
    });

    IpcInferenceConfig config;
    config.socket_path = worker.socket_path();
    config.connect_timeout_seconds = 2;
    config.request_timeout_seconds = 2;

    IpcInferenceBackend backend(config);
    backend.initialize();

    InferenceRequest request = make_request("inspect");
    request.system_message = "Sys prompt.";
    request.use_system_prompt_cache = false;

    ASSERT_TRUE(backend.infer(request).success);
    worker.join_and_rethrow();
  }

  // Cache flag should be set in the first request, not in the second.
  EXPECT_TRUE(seen_cached.header.schema_flags & ipc::kSchemaFlagStructured);
  EXPECT_TRUE(seen_cached.header.schema_flags & ipc::kSchemaFlagSysCache);
  EXPECT_TRUE(seen_uncached.header.schema_flags & ipc::kSchemaFlagStructured);
  EXPECT_FALSE(seen_uncached.header.schema_flags & ipc::kSchemaFlagSysCache);
}

TEST(IpcInferenceBackend, StructuredModeHistoryOnlyNoSystemMessage)
{
  ParsedRequest seen;

  OneClientWorker worker([&](int client_fd) {
    seen = read_full_request(client_fd);
    send_success_response(client_fd, seen.header.request_id, "ok");
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  // No system_message, but history present → structured mode, no SysCache flag.
  InferenceRequest request = make_request("now");
  request.history.push_back(HistoryEntry{"prev user", "prev asst"});

  ASSERT_TRUE(backend.infer(request).success);

  EXPECT_TRUE(seen.header.schema_flags & ipc::kSchemaFlagStructured);
  EXPECT_FALSE(seen.header.schema_flags & ipc::kSchemaFlagSysCache);
  EXPECT_EQ(seen.header.system_bytes, 0U);
  EXPECT_EQ(seen.header.history_count, 1U);
  EXPECT_TRUE(seen.system_message.empty());
  ASSERT_EQ(seen.history.size(), 1U);
  EXPECT_EQ(seen.history[0].first, "prev user");
  EXPECT_EQ(seen.history[0].second, "prev asst");
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, TemporalMetadataRoundTrip)
{
  ParsedRequest seen;

  OneClientWorker worker([&](int client_fd) {
    seen = read_full_request(client_fd);
    send_success_response(client_fd, seen.header.request_id, "ok");
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest request = make_request("temporal");
  request.sequence_type = edge_vlm_ros::TemporalSequenceType::kTemporalImages;
  request.fps = 7.5;
  request.frame_timestamps_sec = {0.0};

  ASSERT_TRUE(backend.infer(request).success);
  EXPECT_EQ(seen.header.sequence_type, 1U);
  EXPECT_TRUE(seen.header.schema_flags & ipc::kSchemaFlagHasFps);
  EXPECT_TRUE(seen.header.schema_flags & ipc::kSchemaFlagHasFrameTimestamps);
  EXPECT_DOUBLE_EQ(seen.header.fps, 7.5);
  ASSERT_EQ(seen.frame_timestamps_sec.size(), 1U);
  EXPECT_DOUBLE_EQ(seen.frame_timestamps_sec[0], 0.0);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, RejectsTemporalMetadataInImagesMode)
{
  OneClientWorker worker([](int client_fd) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ::close(client_fd);
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 1;

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest request = make_request("bad");
  request.fps = 4.0;
  EXPECT_THROW(backend.infer(request), std::runtime_error);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, RejectsTimestampFrameCountMismatch)
{
  OneClientWorker worker([](int client_fd) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ::close(client_fd);
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 1;

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest request = make_request("bad");
  request.sequence_type = edge_vlm_ros::TemporalSequenceType::kTemporalImages;
  request.extra_images.push_back(cv::Mat(4, 5, CV_8UC3, cv::Scalar(1, 2, 3)));
  request.frame_timestamps_sec = {0.0};
  EXPECT_THROW(backend.infer(request), std::runtime_error);
  worker.join_and_rethrow();
}

TEST(IpcInferenceBackend, RejectsFpsTimestampConflict)
{
  OneClientWorker worker([](int client_fd) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ::close(client_fd);
  });

  IpcInferenceConfig config;
  config.socket_path = worker.socket_path();
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 1;

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest request = make_request("bad");
  request.sequence_type = edge_vlm_ros::TemporalSequenceType::kTemporalImages;
  request.extra_images.push_back(cv::Mat(4, 5, CV_8UC3, cv::Scalar(1, 2, 3)));
  request.fps = 10.0;
  request.frame_timestamps_sec = {0.0, 0.2};
  EXPECT_THROW(backend.infer(request), std::runtime_error);
  worker.join_and_rethrow();
}
