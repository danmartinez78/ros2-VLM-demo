# Architecture — cosmos_ros2_video_reasoner

## Overview

`cosmos_ros2_video_reasoner` is a single `ament_cmake` ROS 2 package that
bridges an NVIDIA Cosmos Reason2 TensorRT Edge-LLM VLM engine with a standard
ROS 2 camera topic.  It is designed for deployment on NVIDIA Jetson AGX Thor
with JetPack 7.2 and ROS 2 Jazzy.

---

## Component diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│  ROS 2 Node: cosmos_reasoner                                           │
│                                                                        │
│  ┌─────────────────────────┐      ┌──────────────────────────────────┐ │
│  │ Subscription callback   │      │ Inference worker thread          │ │
│  │ /camera/image_raw       │      │                                  │ │
│  │                         │      │  1. Dequeue frame                │ │
│  │  1. Check ROS timestamp │─────▶│  2. cv_bridge decode             │ │
│  │  2. Compare vs period   │ cv:: │  3. Resize (≤ image_max_width)   │ │
│  │  3. Enqueue if due      │ Mat  │  4. cv::imencode → JPEG bytes    │ │
│  │  4. Drop old if busy    │      │  5. InferenceBackend::infer()    │ │
│  └─────────────────────────┘      │  6. Publish VisionReasoningResult│ │
│                                   └──────────────────────────────────┘ │
│                                              │                          │
│                                  ┌───────────▼──────────────┐          │
│                                  │  InferenceBackend (iface) │          │
│                                  └───────────┬──────────────┘          │
│                              ┌───────────────┴────────────────────┐    │
│                              │                                    │    │
│                  ┌───────────▼──────────────┐  ┌─────────────────▼──┐ │
│                  │ TensorRTEdgeLLMBackend   │  │ FakeInferenceBackend│ │
│                  │ (hardware — Jetson Thor) │  │ (unit tests)        │ │
│                  └──────────────────────────┘  └────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
         │ publish                                      │ subscribe
         ▼                                             ▼
/cosmos/reasoning                            /camera/image_raw
(VisionReasoningResult)                     (sensor_msgs/Image)
```

---

## Threading model

| Thread | Responsibility |
|---|---|
| ROS executor | Runs `image_callback`; checks timestamp, enqueues frame, returns immediately |
| Inference worker | Dequeues frame, calls backend, publishes result |

The queue has depth 1.  When `drop_old_frames = true` and a second frame
arrives before the worker finishes the first, the older pending frame is
discarded.  This prevents unbounded memory growth and keeps the node
responsive to the latest view.

---

## Image handling

**Why JPEG encoding?**

TensorRT Edge-LLM ≤ 0.5.0 exposes
`rt::imageUtils::loadImageFromMemory(data, size)`, which expects a
`stbi`-compatible byte buffer (JPEG, PNG, BMP …).  There is no API to pass a
raw pixel pointer directly to the vision encoder.

The chosen approach is:

1. `cv_bridge` converts the ROS `sensor_msgs/Image` to an OpenCV `cv::Mat`.
2. The mat is optionally resized with `cv::resize` (INTER_AREA) to stay within
   `image_max_width`.
3. `cv::imencode(".jpg", …)` encodes it to an in-memory JPEG buffer.
4. `rt::imageUtils::loadImageFromMemory()` parses the JPEG buffer and returns
   an `rt::imageUtils::ImageData` with an internal CPU tensor.

No temporary files are created or left on disk.

**Tradeoff**: JPEG encoding is lossy and adds ~1–3 ms per frame.  At typical
inference latency of ≥500 ms this overhead is negligible.  If a future
TensorRT Edge-LLM release adds a direct raw-pixel ingestion path, the
`TensorRTEdgeLLMBackend::infer()` method should be updated to skip encoding.

---

## Startup validation

During `initialize()` the backend validates:

1. `llm_engine_dir` is a directory that exists.
2. `multimodal_engine_dir` is a directory that exists.
3. `edge_llm_plugin_path` (if non-empty) refers to an existing file.
4. `dlopen` can load the plugin library.
5. `cudaStreamCreate` succeeds (CUDA runtime is available).
6. `LLMInferenceRuntime` construction succeeds (engines can be deserialized).

If any step fails, `initialize()` throws `std::runtime_error` and the node
shuts down with a clear error message before accepting any camera frames.

---

## Shutdown sequence

1. `stop_worker()` sets `worker_running_ = false`, notifies the condition
   variable, then joins the thread.
2. The worker drains its last pending frame (if any) or exits the wait loop.
3. After the thread joins, the node destructor prints frame counters.
4. `TensorRTEdgeLLMBackend` destructor: `runtime_` is destroyed first, then
   `cudaStreamDestroy`, then `dlclose` releases the plugin DSO.

This order is critical: TensorRT calls plugin destructors when the runtime
goes out of scope, which requires the plugin library DSO to remain mapped.

---

## TensorRT Edge-LLM API surface used

| Header | Symbol |
|---|---|
| `common/trtUtils.h` | `loadEdgellmPluginLib()`, `DlDeleter` |
| `runtime/llmInferenceRuntime.h` | `LLMInferenceRuntime` |
| `runtime/llmRuntimeUtils.h` | `LLMGenerationRequest`, `LLMGenerationResponse`, `Message` |
| `runtime/imageUtils.h` | `imageUtils::loadImageFromMemory()`, `ImageData` |

Tested against TensorRT-Edge-LLM commit `8fe7fe102ee0644b02dfc69afc64ff178101cdae`
(mirror: `straylight-software/TensorRT-Edge-LLM`, version 0.5.0).

---

## Known limitations

* **Sampled images, not native video**: TensorRT Edge-LLM 0.5.0 has no
  native video-stream input path.  The node treats the ROS camera stream as
  a sequence of independent images sampled at `sample_period_seconds`.
* **Single-item queue**: The queue depth is fixed at 1 because the inference
  runtime is stateful (KV cache).  Multiple concurrent requests are not yet
  supported.
* **CompressedImage**: `sensor_msgs/msg/CompressedImage` is not yet
  subscribed.  Remapping `/camera/image_raw` to the compressed topic and using
  `cv_bridge::toCvCopy` with `compressed` transport is the recommended
  workaround until full support is added.
