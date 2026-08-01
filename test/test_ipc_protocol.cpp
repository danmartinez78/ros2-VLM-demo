// Copyright 2025 cosmos_ros2_video_reasoner contributors

#include <gtest/gtest.h>

#include "cosmos_ros2_video_reasoner/ipc_protocol.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

namespace ipc = cosmos_ros2_video_reasoner::ipc;

namespace
{
std::atomic<int> g_sigpipe_count{0};

void sigpipe_handler(int)
{
  g_sigpipe_count.fetch_add(1, std::memory_order_relaxed);
}

class ScopedSigpipeHandler
{
public:
  ScopedSigpipeHandler()
  {
    std::memset(&new_action_, 0, sizeof(new_action_));
    std::memset(&old_action_, 0, sizeof(old_action_));
    new_action_.sa_handler = sigpipe_handler;
    sigemptyset(&new_action_.sa_mask);
    sigaction(SIGPIPE, &new_action_, &old_action_);
  }

  ~ScopedSigpipeHandler()
  {
    sigaction(SIGPIPE, &old_action_, nullptr);
  }

private:
  struct sigaction new_action_{};
  struct sigaction old_action_{};
};
}  // namespace

TEST(IpcProtocol, HeadersUseCurrentProtocolDefaults)
{
  ipc::RequestHeader request;
  ipc::ResponseHeader response;

  EXPECT_EQ(request.magic, ipc::kMagic);
  EXPECT_EQ(request.version, ipc::kVersion);
  EXPECT_EQ(request.encoding, ipc::kEncodingBgr8);

  EXPECT_EQ(response.magic, ipc::kMagic);
  EXPECT_EQ(response.version, ipc::kVersion);
}

TEST(IpcProtocol, ReadAllHandlesFragmentedInput)
{
  int fds[2]{};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  std::vector<uint8_t> payload(4096);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>(i % 251);
  }

  std::thread writer([&]() {
    size_t offset = 0;
    while (offset < payload.size()) {
      const size_t chunk = std::min<size_t>(17, payload.size() - offset);
      const ssize_t written = ::write(fds[1], payload.data() + offset, chunk);
      if (written <= 0) {
        break;
      }
      offset += static_cast<size_t>(written);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ::close(fds[1]);
  });

  std::vector<uint8_t> received(payload.size(), 0);
  ASSERT_NO_THROW(ipc::read_all(fds[0], received.data(), received.size()));
  EXPECT_EQ(received, payload);

  writer.join();
  ::close(fds[0]);
}

TEST(IpcProtocol, WriteAllHandlesSlowReader)
{
  int fds[2]{};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  int send_buffer = 4096;
  ASSERT_EQ(
    ::setsockopt(fds[0], SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer)),
    0);

  std::vector<uint8_t> payload(128U * 1024U, 0x5A);
  std::vector<uint8_t> received;
  received.reserve(payload.size());

  std::thread reader([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::vector<uint8_t> chunk(2048);
    while (received.size() < payload.size()) {
      const ssize_t n = ::read(fds[1], chunk.data(), chunk.size());
      if (n <= 0) {
        break;
      }
      received.insert(received.end(), chunk.begin(), chunk.begin() + n);
    }
    ::close(fds[1]);
  });

  ASSERT_NO_THROW(ipc::write_all(fds[0], payload.data(), payload.size()));
  ::shutdown(fds[0], SHUT_WR);

  reader.join();
  EXPECT_EQ(received, payload);

  ::close(fds[0]);
}

TEST(IpcProtocol, ReadAllFailsCleanlyOnTruncatedPayload)
{
  int fds[2]{};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  std::vector<uint8_t> partial(128, 0xA5);
  ASSERT_EQ(::write(fds[1], partial.data(), partial.size()), static_cast<ssize_t>(partial.size()));
  ::close(fds[1]);

  std::vector<uint8_t> destination(256, 0);
  EXPECT_THROW(ipc::read_all(fds[0], destination.data(), destination.size()), std::runtime_error);

  ::close(fds[0]);
}

TEST(IpcProtocol, WriteAllDoesNotRaiseSigpipeWhenPeerClosed)
{
  ScopedSigpipeHandler scoped_handler;
  g_sigpipe_count.store(0, std::memory_order_relaxed);

  int fds[2]{};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  ::close(fds[1]);

  std::vector<uint8_t> payload(128, 0x7F);
  EXPECT_THROW(ipc::write_all(fds[0], payload.data(), payload.size()), std::runtime_error);
  EXPECT_EQ(g_sigpipe_count.load(std::memory_order_relaxed), 0);

  ::close(fds[0]);
}

TEST(IpcProtocol, RequestHeaderV2FieldsDefaultToInlineMode)
{
  ipc::RequestHeader header;

  // Schema v2 request header defaults must match inline (backward-compatible) delivery.
  EXPECT_EQ(header.version, ipc::kVersion);
  EXPECT_EQ(header.schema_flags, ipc::kSchemaFlagInline);
  EXPECT_EQ(header.system_bytes, 0U);
  EXPECT_EQ(header.history_count, 0U);

  // Verify the structured and cache flags are distinct and non-zero.
  EXPECT_NE(ipc::kSchemaFlagStructured, ipc::kSchemaFlagInline);
  EXPECT_NE(ipc::kSchemaFlagSysCache, ipc::kSchemaFlagInline);
  EXPECT_EQ(ipc::kSchemaFlagStructured & ipc::kSchemaFlagSysCache, 0U);
}

TEST(IpcProtocol, HistoryEntryHeaderDefaultsToZero)
{
  ipc::HistoryEntryHeader entry{};
  EXPECT_EQ(entry.user_bytes, 0U);
  EXPECT_EQ(entry.asst_bytes, 0U);
}
