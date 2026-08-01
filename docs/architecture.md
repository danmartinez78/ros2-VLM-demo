# Architecture

## Purpose

`cosmos_ros2_video_reasoner` connects a ROS 2 Jazzy raw-image topic to a
persistent NVIDIA Cosmos Reason2 TensorRT Edge-LLM runtime on Jetson AGX Thor.
It publishes a structured `VisionReasoningResult` for each sampled frame.

The production design uses two processes. This is a correctness requirement on
the validated Thor stack, not merely a deployment preference.

## Process topology

```mermaid
flowchart TD
    SOURCE[ROS bag or live camera] -->|sensor_msgs/Image| NODE
    subgraph RP[ROS process]
      NODE[cosmos_reasoner]
      QUEUE[Newest-frame queue]
      IPC[IpcInferenceBackend]
      NODE --> QUEUE --> IPC
    end
    IPC <-->|versioned Unix socket| SERVER
    subgraph GP[ROS-free GPU process]
      SERVER[cosmos_inference_worker]
      BACKEND[TensorRTEdgeLLMBackend]
      ENGINE[Persistent LLM and visual engines]
      SERVER --> BACKEND --> ENGINE
    end
    NODE -->|VisionReasoningResult| RESULT[/cosmos/reasoning]
```

### `cosmos_reasoner`

The ROS process:

- subscribes to a raw `sensor_msgs/msg/Image` topic;
- samples by message timestamp;
- validates and converts `bgr8`, `rgb8`, or `mono8` to packed BGR8;
- resizes frames while preserving aspect ratio;
- keeps at most one pending frame;
- sends requests to the worker over a Unix-domain socket;
- publishes results and failure information.

It does not link CUDA, TensorRT, Edge-LLM, RMW-specific GPU code, or
`libcosmos_trt_backend.so`.

### `cosmos_inference_worker`

The worker process:

- does not link ROS, RMW, DDS, or `cv_bridge`;
- loads the Edge-LLM plugin, Cosmos engines, tokenizer, and visual engine once;
- owns the CUDA context, non-blocking CUDA stream, and TensorRT runtime;
- converts received BGR8 frames to in-memory JPEG;
- calls `LLMInferenceRuntime::handleRequest()` serially;
- returns text, error state, and measured inference duration.

The final worker executable directly links the Edge-LLM core, CuTe DSL AOT
archive, TensorRT, and CUDA. This ensures the Thor `sm_110a` device-link step
occurs at the final executable boundary.

## Why process isolation is required

During Thor bring-up, native NVIDIA inference succeeded with the same engine
and image while in-process ROS integrations stalled indefinitely in a
TensorRT Blackwell fused-attention prefill kernel. The following controls were
important:

- direct-linked Edge-LLM without ROS libraries: succeeded;
- the same direct-linked code on a `std::thread`: succeeded;
- the direct-linked executable merely linked to `rclcpp`, without calling
  `rclcpp::init()`: reproduced the stall;
- ROS and Edge-LLM separated into different processes: succeeded repeatedly.

This localizes the trigger to co-loading the ROS 2 transitive native dependency
set with Edge-LLM on the validated platform. The exact conflicting shared
object or symbol interaction has not been identified. The process boundary is
therefore the supported production architecture.

See [thor-edge-llm-prefill-stall-rca.md](thor-edge-llm-prefill-stall-rca.md)
for the evidence and discarded hypotheses.

## ROS threading and queueing

| Execution context | Responsibility |
| --- | --- |
| ROS executor thread | Timestamp sampling and newest-frame enqueue |
| ROS inference thread | Image conversion, resize, synchronous IPC, publication |
| Worker main thread | Request validation and TensorRT inference |
| TensorRT/CUDA streams | Visual preprocessing, prefill, and generation |

The queue depth is one. If inference is busy and another sampled frame arrives,
the pending frame is replaced when `drop_old_frames` is enabled. The in-flight
frame is never cancelled. This avoids unbounded memory use and prevents stale
video frames from accumulating.

## Image path

1. ROS receives `sensor_msgs/msg/Image`.
2. The node validates width, height, row stride, payload size, and encoding.
3. It converts to a packed `CV_8UC3` BGR image without `cv_bridge`.
4. It optionally resizes to `image_max_width`.
5. IPC transfers the packed BGR bytes and dimensions.
6. The worker encodes JPEG in memory using configured `jpeg_quality`.
7. Edge-LLM parses the JPEG with `loadImageFromMemory()`.
8. The image buffer is moved into the generation request.

No per-frame temporary file is used.

Avoiding `cv_bridge` is deliberate. The tested Thor host exposed ROS OpenCV 4.6
through `libcv_bridge.so` and NVIDIA OpenCV 4.8 through JetPack. Loading both
C++ ABIs into the production process is unnecessary and unsafe.

## IPC protocol

The socket defaults to `/tmp/cosmos_edge_llm.sock`. Requests and responses use
fixed, trivially-copyable headers followed by bounded byte payloads.
The current schema version is **2** (`kVersion = 2`).

### Request (v2)

| Field group | Contents |
| --- | --- |
| Identity | magic, protocol version (2), monotonically increasing request ID |
| Image | encoding ID, width, height, packed step, byte length, BGR8 bytes |
| Schema | `schema_flags` (delivery mode bits), `system_bytes`, `history_count` |
| Task | user-message text length and bytes |
| Generation | maximum tokens, temperature, top-p, top-k |

#### Delivery modes (`schema_flags`)

| Value | Constant | Wire layout after the image bytes |
| --- | --- | --- |
| `0` | `kSchemaFlagInline` | `[prompt_bytes bytes]` — legacy inline delivery |
| `1` | `kSchemaFlagStructured` | `[system_bytes bytes][prompt_bytes bytes][history_count × entry]` |
| `3` | `kSchemaFlagStructured \| kSchemaFlagSysCache` | structured + request system-prompt caching |

Each history entry is preceded by a `HistoryEntryHeader` containing
`user_bytes` and `asst_bytes`, followed immediately by the user text and
assistant text bytes:
```
[HistoryEntryHeader][user_bytes bytes][asst_bytes bytes]
```

Prior assistant outputs are carried as untrusted observations in the history
user/assistant turn pairs. They are **never** promoted to system-role authority.

#### System-prompt cache (`kSchemaFlagSysCache`)

When `kSchemaFlagSysCache` is set, the worker may attempt system-prompt caching
for the associated system message using the TensorRT Edge-LLM runtime API.
Cache eligibility rules:

- The flag is only meaningful when `kSchemaFlagStructured` is also set and
  `system_bytes > 0`.
- Caching is only attempted for exact, stable system prompt text.  Any change
  to the system prompt invalidates the cache key.
- Cache keys are in-memory only and do not survive worker restart.
- Multimodal (image-containing) system prompts are not cache-eligible.
- The flag is **silently ignored** when the runtime or model does not support
  the feature.  Always falls back to uncached delivery.
- Enable via `enable_system_prompt_cache: true` in `cosmos_reasoner.yaml`, only
  valid together with `instruction_delivery_mode: structured`.

> **Thor validation required**: System-prompt caching has not been benchmarked
> on the validated Jetson AGX Thor stack.  Enable it only after measuring
> cache-hit TTFT using NVIDIA native profiling and the methodology from
> issue #7.  To measure: run `cosmos_inference_worker --benchmark-session` with
> the cache enabled and disabled across a representative prompt set.

### Response

| Field group | Contents |
| --- | --- |
| Identity | magic, protocol version, matching request ID |
| Result | success flag and inference duration |
| Payload | bounded response text and error text |

Limits are enforced before allocation or transfer:

- image payload: 256 MiB;
- prompt, system message, history entry, response, and error text: 1 MiB each;
- history entries: 256 maximum;
- socket path: `sockaddr_un::sun_path` capacity.

IPC writes suppress `SIGPIPE`. A transport failure closes the client socket.
The current frame fails once; the next sampled frame attempts a new connection.
Requests are not automatically replayed because execution state is uncertain
after a connection failure.

The protocol uses native POD layout and is intended only for the two binaries
built from the same package on one host. It is not a network API or a
cross-version serialization format.

## Startup

Launch starts the worker and ROS process together:

1. worker creates the Unix socket;
2. worker loads the plugin and initializes Edge-LLM;
3. worker begins listening;
4. ROS inference thread connects, with a 120-second default deadline;
5. only after backend initialization succeeds does the node create its camera
   subscription;
6. rosbag test scripts wait for that subscription before playback.

This prevents frames from being consumed before the engines are ready.

The missing optional action engine message is informational for image-only
Cosmos Reason2 deployments and does not fail initialization.

## Failure and recovery behavior

| Failure | Behavior |
| --- | --- |
| Worker fails during startup | ROS initialization fails with a connection timeout |
| Worker exits after startup | Launch respawns it after two seconds |
| Socket read/write fails | Current frame fails; client disconnects |
| Replacement worker becomes ready | Next sampled frame reconnects |
| Invalid response | Connection is discarded and frame fails |
| Worker is alive but GPU call is wedged | Worker-side watchdog fires after `worker_inference_deadline_seconds`; worker emits diagnostic and calls `std::_Exit`; client sees EOF and reports one error; launch respawns the worker |

Crash recovery has been validated by killing the worker with `SIGKILL` during
looping rosbag playback. The ROS node survived, launch created a new worker PID,
and inference resumed.

## Watchdog recovery for a wedged worker

When the TensorRT call inside the worker does not return within
`worker_inference_deadline_seconds` (default 60 s), a watchdog thread:

1. emits a structured diagnostic to `stderr`:
   ```
   [cosmos_inference_worker] WATCHDOG: inference deadline (60s) expired request_id=N; self-terminating for clean respawn
   ```
2. calls `std::_Exit(1)`, which bypasses **all** C++ destructors, `std::atexit`
   handlers, and `std::at_quick_exit` handlers. The OS reclaims all file
   descriptors immediately.

The socket file is not removed by `_Exit`. The replacement worker calls
`::unlink()` at startup before creating the new listener socket.

### Process-termination primitive: `_Exit` vs `quick_exit`

`std::_Exit` is used instead of `std::quick_exit` for two reasons:

1. **`quick_exit` still invokes `at_quick_exit` handlers.** If the CUDA runtime
   or TensorRT registers an `at_quick_exit` handler (which is permitted by the
   SDK), that handler could attempt to tear down the wedged CUDA context and
   block indefinitely — exactly the hang we are trying to escape.

2. **`_Exit` provides a hard process boundary.** No third-party cleanup code
   runs; the OS reclaims all resources atomically. This is the correct isolation
   primitive when the only known-safe action is to exit and let the process
   supervisor restart a fresh worker.

### Deadline relationship

```
worker_inference_deadline_seconds  <  worker_request_timeout_seconds
       (default: 60 s)                       (default: 90 s)
```

The 30-second gap gives the worker time to print the diagnostic and exit before
the client-side `SO_RCVTIMEO` fires. When the worker exits cleanly, the client
sees an EOF (`IPC peer closed`) rather than a socket timeout. Either error path
closes the client connection; only one error is reported for the timed-out
request, which is not automatically replayed.

The deadline constraint is validated at two levels:
- **Launch time**: `cosmos_reasoner.launch.py` raises a `RuntimeError` before
  starting either process if `worker_inference_deadline_seconds >= worker_request_timeout_seconds`.
- **Node startup**: `cosmos_reasoner` logs `FATAL` and exits if the constraint
  is violated (defense-in-depth for non-launch invocations).

### TensorRT Edge-LLM cancellation API

The TensorRT Edge-LLM SDK in the pinned version does not expose a supported
request-cancellation or in-flight deadline API for `LLMInferenceRuntime::handleRequest()`.
Process termination via `_Exit` is therefore the isolation mechanism. This note
should be revisited if a future SDK version adds cancellation support.

## Shutdown

ROS launch owns both child processes. Normal launch shutdown signals the worker
and node, then removes the worker socket. The ROS node stops accepting new
frames, wakes its inference thread, and joins it before destruction.

A CUDA call already executing in the worker may not respond promptly to a
normal signal. Because it is a separate process, it can be escalated to
`SIGKILL` without wedging ROS shutdown.

If the worker self-terminates via the inference deadline watchdog, the socket
file is left in place until the replacement worker removes it at startup.
Launch respawns the worker after `respawn_delay` (default 2 s), then the
replacement worker creates a new socket and becomes ready for the next sampled
frame.

## Build boundaries

Production targets:

| Target | Role |
| --- | --- |
| `cosmos_reasoner_node` | Hardware-independent ROS node library |
| `cosmos_ipc_backend` | ROS-side Unix-socket client |
| `cosmos_reasoner` | ROS executable; IPC only |
| `cosmos_inference_worker` | ROS-free, direct-linked GPU executable |

Diagnostic targets retained for RCA and future compatibility work:

| Target | Role |
| --- | --- |
| `cosmos_trt_backend` | Shared Edge-LLM backend that reproduced the failing boundary; not used by production launch |
| `cosmos_backend_direct_smoke` | Standalone diagnostic for link and ROS-library-load experiments |

Deployment verification checks dynamic dependencies to ensure the production
process boundary has not regressed.

## Known limitations and follow-ups

- Independent frames rather than temporal video windows: issue #8.
- Formal latency/resource benchmarks: issue #7.
- Model portability and measured optimization: issue #9.
- RViz2 visualization: issue #10.
- Task-level quality evaluation: issue #11.
- System-prompt cache Thor benchmark: requires measuring cache-hit TTFT on the
  validated stack before enabling `enable_system_prompt_cache` in production
  (see IPC protocol section above).
