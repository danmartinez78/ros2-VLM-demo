// Copyright 2025 cosmos_ros2_video_reasoner contributors

#include <gtest/gtest.h>

#include "cosmos_ros2_video_reasoner/ipc_inference_backend.hpp"
#include "cosmos_ros2_video_reasoner/ipc_protocol.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace
{
using cosmos_ros2_video_reasoner::InferenceRequest;
using cosmos_ros2_video_reasoner::IpcInferenceBackend;
using cosmos_ros2_video_reasoner::IpcInferenceConfig;
namespace ipc = cosmos_ros2_video_reasoner::ipc;

std::atomic<uint64_t> g_path_counter{0};

std::string make_socket_path()
{
  const uint64_t id = g_path_counter.fetch_add(1, std::memory_order_relaxed);
  return "/tmp/cosmos-ipc-test-" + std::to_string(static_cast<long long>(::getpid())) + "-" +
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
    if (ready_future_.wait_for(std::chrono::seconds(5)) != std::future_status::ready ||
      !ready_future_.get())
    {
      throw std::runtime_error("failed to start test worker");
    }
  }

  ~OneClientWorker()
  {
    if (thread_.joinable()) {
      thread_.join();
    }
    ::unlink(path_.c_str());
  }

  std::string const & socket_path() const {return path_;}

private:
  void run()
  {
    ::unlink(path_.c_str());
    int server_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (server_fd < 0) {
      ready_promise_.set_value(false);
      return;
    }

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strncpy(address.sun_path, path_.c_str(), sizeof(address.sun_path) - 1);

    if (::bind(server_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
      ::listen(server_fd, 1) != 0)
    {
      ::close(server_fd);
      ready_promise_.set_value(false);
      return;
    }

    ready_promise_.set_value(true);

    int client_fd = ::accept4(server_fd, nullptr, nullptr, SOCK_CLOEXEC);
    if (client_fd >= 0) {
      handler_(client_fd);
      ::close(client_fd);
    }

    ::close(server_fd);
  }

  std::string path_;
  SessionHandler handler_;
  std::thread thread_;
  std::promise<bool> ready_promise_;
  std::future<bool> ready_future_{ready_promise_.get_future()};
};

void read_request_frame(int fd, ipc::RequestHeader & header, std::vector<uint8_t> & image, std::string & prompt)
{
  ipc::read_all(fd, &header, sizeof(header));
  image.resize(header.image_bytes);
  prompt.resize(header.prompt_bytes);
  ipc::read_all(fd, image.data(), image.size());
  ipc::read_all(fd, prompt.data(), prompt.size());
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

  IpcInferenceBackend backend(config);
  backend.initialize();

  InferenceRequest oversized_prompt = make_request();
  oversized_prompt.prompt.assign(ipc::kMaxTextBytes + 1U, 'x');
  EXPECT_THROW(backend.infer(oversized_prompt), std::runtime_error);

  std::vector<uint8_t> one_byte(1, 0);
  InferenceRequest oversized_image = make_request();
  oversized_image.image = cv::Mat(
    1,
    static_cast<int>(ipc::kMaxImageBytes / 3U + 1U),
    CV_8UC3,
    one_byte.data());
  EXPECT_THROW(backend.infer(oversized_image), std::runtime_error);
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

  EXPECT_LT(elapsed, std::chrono::seconds(3));
}

TEST(IpcInferenceBackend, ReconnectsAfterWorkerRestart)
{
  std::string socket_path = make_socket_path();

  OneClientWorker worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);
    send_success_response(client_fd, request_header.request_id, "first");
    ::shutdown(client_fd, SHUT_RDWR);
  }, socket_path);

  IpcInferenceConfig config;
  config.socket_path = socket_path;
  config.connect_timeout_seconds = 2;
  config.request_timeout_seconds = 2;

  IpcInferenceBackend backend(config);
  backend.initialize();

  auto first = backend.infer(make_request("first request"));
  ASSERT_TRUE(first.success);
  EXPECT_EQ(first.text, "first");

  EXPECT_THROW(backend.infer(make_request("expected disconnect")), std::runtime_error);

  OneClientWorker restarted_worker([&](int client_fd) {
    ipc::RequestHeader request_header;
    std::vector<uint8_t> image;
    std::string prompt;
    read_request_frame(client_fd, request_header, image, prompt);
    send_success_response(client_fd, request_header.request_id, "second");
  }, socket_path);

  auto second = backend.infer(make_request("after restart"));
  EXPECT_TRUE(second.success);
  EXPECT_EQ(second.text, "second");
}
