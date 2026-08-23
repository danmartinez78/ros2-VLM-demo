// Copyright 2025 edge_vlm_ros contributors
#include "edge_vlm_ros/inference_watchdog.hpp"
#include "edge_vlm_ros/ipc_protocol.hpp"
#include "edge_vlm_ros/tensorrt_edge_llm_backend.hpp"

#include <opencv2/core.hpp>

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace
{
volatile std::sig_atomic_t g_stop = 0;
volatile std::sig_atomic_t g_server_fd = -1;
volatile std::sig_atomic_t g_client_fd = -1;

void stop_handler(int)
{
  g_stop = 1;
  if (g_client_fd >= 0) {
    ::close(static_cast<int>(g_client_fd));
    g_client_fd = -1;
  }
  if (g_server_fd >= 0) {
    ::close(static_cast<int>(g_server_fd));
    g_server_fd = -1;
  }
}
}

int main(int argc, char ** argv)
{
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <llm-engine-dir> <multimodal-engine-dir> <plugin-path> <socket-path>"
              << " <jpeg-quality> <inference-deadline-seconds>\n";
    return 2;
  }

  const std::string socket_path = argv[4];
  int jpeg_quality = 0;
  try {
    jpeg_quality = std::stoi(argv[5]);
  } catch (std::exception const &) {
    std::cerr << "JPEG quality must be an integer in [1, 100]\n";
    return 2;
  }
  if (jpeg_quality < 1 || jpeg_quality > 100) {
    std::cerr << "JPEG quality must be in [1, 100]\n";
    return 2;
  }

  int inference_deadline_seconds = 0;
  try {
    inference_deadline_seconds = std::stoi(argv[6]);
  } catch (std::exception const &) {
    std::cerr << "inference-deadline-seconds must be a positive integer\n";
    return 2;
  }
  if (inference_deadline_seconds <= 0) {
    std::cerr << "inference-deadline-seconds must be > 0\n";
    return 2;
  }
  if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
    std::cerr << "Socket path is too long\n";
    return 2;
  }

  std::signal(SIGINT, stop_handler);
  std::signal(SIGTERM, stop_handler);
  ::unlink(socket_path.c_str());

  int server_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  g_server_fd = server_fd;
  if (server_fd < 0) {
    std::cerr << "Could not create worker socket\n";
    return 3;
  }

  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, socket_path.c_str(), sizeof(address.sun_path) - 1);
  if (::bind(server_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
    ::listen(server_fd, 1) != 0)
  {
    std::cerr << "Could not bind/listen on " << socket_path << ": " << std::strerror(errno) << '\n';
    ::close(server_fd);
    return 3;
  }

  try {
    edge_vlm_ros::TensorRTEdgeLLMConfig config;
    config.llm_engine_dir = argv[1];
    config.multimodal_engine_dir = argv[2];
    config.edge_llm_plugin_path = argv[3];
    config.jpeg_quality = jpeg_quality;
    edge_vlm_ros::TensorRTEdgeLLMBackend backend(config);
    backend.initialize();
    std::cout << "edge_vlm_server ready on " << socket_path << std::endl;

    // ── Test-only one-shot injected hang ─────────────────────────────────
    // EDGE_VLM_TEST_INJECT_HANG_ONCE_SENTINEL names a sentinel file path.
    // The first inference request will atomically create the sentinel and
    // then sleep past the watchdog deadline so the watchdog fires and calls
    // std::_Exit(1).  Subsequent workers find the sentinel already present
    // and skip the hang, allowing normal inference to proceed.
    //
    // DISABLED by default (env var unset).  Set only for hardware recovery
    // validation — never in production.
    const char * const test_sentinel_path =
      std::getenv("EDGE_VLM_TEST_INJECT_HANG_ONCE_SENTINEL");

    // The service outlives individual clients. This permits a CLI, experiment
    // harness, and ROS adapter to connect sequentially without reloading engines.
    while (!g_stop) {
      int client_fd = ::accept4(server_fd, nullptr, nullptr, SOCK_CLOEXEC);
      if (client_fd < 0) {
        if (g_stop || errno == EINTR || errno == EBADF) {
          break;
        }
        throw std::runtime_error(
                "worker accept failed: " + std::string(std::strerror(errno)));
      }
      g_client_fd = client_fd;

      while (!g_stop) {
        edge_vlm_ros::ipc::RequestHeader header;
        try {
          edge_vlm_ros::ipc::read_all(client_fd, &header, sizeof(header));
        } catch (std::exception const &) {
          break;
        }
        if (header.magic != edge_vlm_ros::ipc::kMagic ||
          header.version != edge_vlm_ros::ipc::kVersion ||
          header.encoding != edge_vlm_ros::ipc::kEncodingBgr8 ||
          header.width == 0 || header.height == 0 ||
          header.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
          header.height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
          static_cast<uint64_t>(header.step) !=
          static_cast<uint64_t>(header.width) * 3U ||
          header.prompt_bytes > edge_vlm_ros::ipc::kMaxTextBytes ||
          header.system_bytes > edge_vlm_ros::ipc::kMaxTextBytes ||
          header.history_count > edge_vlm_ros::ipc::kMaxHistoryEntries)
        {
          throw std::runtime_error("invalid IPC request header");
        }
        const uint64_t expected_image_bytes =
          static_cast<uint64_t>(header.step) * header.height;
        if (expected_image_bytes != header.image_bytes ||
          header.image_bytes > edge_vlm_ros::ipc::kMaxImageBytes)
        {
          throw std::runtime_error("invalid IPC image payload size");
        }

        // ── Multi-image: validate and read PerImageHeaders ────────────────────
        const bool is_multi_image =
          (header.schema_flags & edge_vlm_ros::ipc::kSchemaFlagMultiImage) != 0;
        const uint32_t image_count = is_multi_image ? header.image_count : 1U;
        if (is_multi_image &&
          (image_count < 2U || image_count > edge_vlm_ros::ipc::kMaxExtraImages + 1U))
        {
          throw std::runtime_error("invalid IPC multi-image count");
        }
        struct ExtraImageInfo
        {
          uint32_t width;
          uint32_t height;
          uint32_t step;
          uint32_t image_bytes;
        };
        std::vector<ExtraImageInfo> extra_infos;
        if (is_multi_image) {
          extra_infos.reserve(image_count - 1U);
          for (uint32_t ei = 0; ei < image_count - 1U; ++ei) {
            edge_vlm_ros::ipc::PerImageHeader pih;
            edge_vlm_ros::ipc::read_all(client_fd, &pih, sizeof(pih));
            if (pih.width == 0 || pih.height == 0 ||
              static_cast<uint64_t>(pih.step) != static_cast<uint64_t>(pih.width) * 3U ||
              static_cast<uint64_t>(pih.step) * pih.height != pih.image_bytes ||
              pih.image_bytes > edge_vlm_ros::ipc::kMaxImageBytes)
            {
              throw std::runtime_error("invalid IPC extra image header");
            }
            extra_infos.push_back({pih.width, pih.height, pih.step, pih.image_bytes});
          }
        }

        // ── Read primary image ────────────────────────────────────────────────
        std::vector<uint8_t> image_bytes(header.image_bytes);
        edge_vlm_ros::ipc::read_all(
          client_fd, image_bytes.data(), image_bytes.size());

        // ── Read extra images (multi-image path) ──────────────────────────────
        std::vector<std::vector<uint8_t>> extra_image_bytes;
        if (is_multi_image) {
          extra_image_bytes.reserve(extra_infos.size());
          for (const auto & info : extra_infos) {
            std::vector<uint8_t> buf(info.image_bytes);
            edge_vlm_ros::ipc::read_all(client_fd, buf.data(), buf.size());
            extra_image_bytes.push_back(std::move(buf));
          }
        }

        // ── Read structured or inline payload ─────────────────────────────────
        const bool is_structured =
          (header.schema_flags & edge_vlm_ros::ipc::kSchemaFlagStructured) != 0;
  
        std::string system_message;
        std::string prompt;
        std::vector<edge_vlm_ros::HistoryEntry> history;
  
        if (is_structured) {
          // Read system message (may be empty).
          if (header.system_bytes > 0) {
            system_message.resize(header.system_bytes);
            edge_vlm_ros::ipc::read_all(
              client_fd, system_message.data(), system_message.size());
          }
          // Read user message.
          prompt.resize(header.prompt_bytes);
          edge_vlm_ros::ipc::read_all(client_fd, prompt.data(), prompt.size());
          // Read history entries.
          history.resize(header.history_count);
          for (auto & entry : history) {
            edge_vlm_ros::ipc::HistoryEntryHeader entry_header;
            edge_vlm_ros::ipc::read_all(
              client_fd, &entry_header, sizeof(entry_header));
            if (entry_header.user_bytes > edge_vlm_ros::ipc::kMaxTextBytes ||
              entry_header.asst_bytes > edge_vlm_ros::ipc::kMaxTextBytes)
            {
              throw std::runtime_error("IPC history entry exceeds protocol limits");
            }
            entry.user_text.resize(entry_header.user_bytes);
            entry.asst_text.resize(entry_header.asst_bytes);
            edge_vlm_ros::ipc::read_all(
              client_fd, entry.user_text.data(), entry.user_text.size());
            edge_vlm_ros::ipc::read_all(
              client_fd, entry.asst_text.data(), entry.asst_text.size());
          }
        } else {
          // Inline mode: single prompt string, no system message, no history.
          prompt.resize(header.prompt_bytes);
          edge_vlm_ros::ipc::read_all(client_fd, prompt.data(), prompt.size());
        }
  
        cv::Mat view(
          static_cast<int>(header.height), static_cast<int>(header.width),
          CV_8UC3, image_bytes.data(), header.step);
  
        edge_vlm_ros::InferenceRequest request;
        request.image = view;
        if (is_multi_image) {
          request.extra_images.reserve(extra_infos.size());
          for (size_t ei = 0; ei < extra_infos.size(); ++ei) {
            const auto & info = extra_infos[ei];
            cv::Mat extra_view(
              static_cast<int>(info.height), static_cast<int>(info.width),
              CV_8UC3, extra_image_bytes[ei].data(), info.step);
            request.extra_images.push_back(extra_view.clone());
          }
        }
        request.prompt = std::move(prompt);
        request.system_message = std::move(system_message);
        request.history = std::move(history);
        request.use_system_prompt_cache =
          (header.schema_flags & edge_vlm_ros::ipc::kSchemaFlagSysCache) != 0;
        request.max_generate_length = header.max_generate_length;
        request.temperature = header.temperature;
        request.top_p = header.top_p;
        request.top_k = header.top_k;
  
        // ── Test-only: check whether to inject a one-shot hang ───────────────
        // O_CREAT|O_EXCL is atomic: only the first worker to process a request
        // succeeds in creating the sentinel.  All later workers (or later
        // requests from the same worker) find the file already present and
        // proceed normally.  The sleep below exceeds the watchdog deadline so
        // the watchdog fires and calls std::_Exit(1) before the sleep ends.
        bool inject_hang = false;
        if (test_sentinel_path != nullptr && *test_sentinel_path != '\0') {
          int sfd = ::open(test_sentinel_path, O_CREAT | O_EXCL | O_WRONLY, 0600);
          if (sfd >= 0) {
            ::close(sfd);
            inject_hang = true;
          }
        }
  
        // ── Worker-side inference deadline watchdog ──────────────────────────
        // Guards the TensorRT call with a configurable deadline. If the call
        // wedges past the deadline, the watchdog emits a diagnostic and calls
        // std::_Exit(1) — see watchdog_exit_on_expire for the rationale for
        // _Exit over quick_exit.
        edge_vlm_ros::InferenceWatchdog watchdog(
          inference_deadline_seconds, header.request_id,
          edge_vlm_ros::watchdog_exit_on_expire);
  
        edge_vlm_ros::InferenceResponse result;
        if (inject_hang) {
          // Intentionally sleep past the watchdog deadline so the watchdog
          // fires and calls std::_Exit(1).  The sleep_for call is unreachable
          // past _Exit(1) — the lines after are defensive dead code only.
          std::this_thread::sleep_for(
            std::chrono::seconds(inference_deadline_seconds + 30));
          result.success = false;
          result.error = "injected hang (unreachable after watchdog)";
        } else {
          try {
            result = backend.infer(request);
          } catch (std::exception const & error) {
            result.success = false;
            result.error = error.what();
          }
        }
  
        // Signal the watchdog that inference completed within the deadline.
        watchdog.cancel();
  
        if (result.text.size() > edge_vlm_ros::ipc::kMaxTextBytes ||
          result.error.size() > edge_vlm_ros::ipc::kMaxTextBytes)
        {
          throw std::runtime_error("inference response exceeds IPC protocol limits");
        }
  
        edge_vlm_ros::ipc::ResponseHeader response;
        response.request_id = header.request_id;
        response.success = result.success ? 1U : 0U;
        response.text_bytes = static_cast<uint32_t>(result.text.size());
        response.error_bytes = static_cast<uint32_t>(result.error.size());
        response.inference_seconds = result.inference_seconds;
        edge_vlm_ros::ipc::write_all(client_fd, &response, sizeof(response));
        edge_vlm_ros::ipc::write_all(client_fd, result.text.data(), result.text.size());
        edge_vlm_ros::ipc::write_all(
          client_fd, result.error.data(), result.error.size());
      }
      ::close(client_fd);
      g_client_fd = -1;
    }
  } catch (std::exception const & error) {
    std::cerr << "Inference worker failed: " << error.what() << '\n';
    if (g_client_fd >= 0) {
      ::close(static_cast<int>(g_client_fd));
      g_client_fd = -1;
    }
    if (g_server_fd >= 0) {
      ::close(static_cast<int>(g_server_fd));
      g_server_fd = -1;
    }
    ::unlink(socket_path.c_str());
    return 4;
  }

  if (g_server_fd >= 0) {
    ::close(static_cast<int>(g_server_fd));
    g_server_fd = -1;
  }
  ::unlink(socket_path.c_str());
  return 0;
}
