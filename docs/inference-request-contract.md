# Inference request contract

This document describes exactly how inference is invoked in `edge_vlm_ros`, from the logical request constructed by a client through the versioned Unix-socket IPC protocol and into the two current worker implementations:

- the **TensorRT Edge-LLM worker** (`edge_vlm_server` / `TensorRTEdgeLLMBackend`); and
- the **FlashRT Cosmos3 worker** (`flashrt_ipc_worker.py`).

The important architectural point is that both workers accept the same IPC v3 request contract, but they do not interpret every field identically. A caller should therefore distinguish between:

1. the **logical request contract** shared by the pipeline;
2. the **wire representation** used to transport that request; and
3. the **runtime-specific mapping** performed by each worker.

## End-to-end request path

```text
ROS node / CLI / benchmark harness
        |
        | logical InferenceRequest
        v
IPC client
        |
        | IPC protocol v3 over Unix domain socket
        | RequestHeader + frame bytes + optional timing/context payloads
        v
Inference worker
        |
        | runtime-specific message/media construction
        v
Model processor / VLM runtime
        |
        v
ResponseHeader + generated text / error / timing / representation provenance
```

The IPC boundary is intentionally model-neutral. The same request can be received by either worker, while each worker is responsible for translating the generic request into the native structures expected by its VLM runtime.

---

## 1. Logical inference request

The C++ representation of a request is `edge_vlm_ros::InferenceRequest`. The Python FlashRT worker reconstructs the same information from IPC into its `ParsedRequest` structure.

A logical request contains the following groups of information.

### Media

| Field | Meaning |
| --- | --- |
| `image` | Primary image / frame, always frame index 0. |
| `extra_images` | Optional additional frames, in caller-defined order. Together with `image`, these form the complete ordered frame list. |
| `sequence_type` | Declares how the caller intends the ordered frame list to be interpreted: independent images, temporal images, or video. |
| `fps` | Optional nominal frame rate for temporal/video interpretation. |
| `frame_timestamps_sec` | Optional exact timestamp for every frame. Values must be finite and strictly increasing. |

All IPC media is transported as packed **BGR8**. A multi-frame request may carry at most 32 total frames: one primary frame plus at most 31 extra frames.

### Prompt and message context

| Field | Meaning |
| --- | --- |
| `prompt` | Current user/task text for this inference. In inline mode this can contain the complete rendered instruction. In structured mode it is the current user-role content. |
| `system_message` | Optional system-role instruction. Presence of a system message causes structured IPC delivery. |
| `history` | Optional ordered prior `(user, assistant)` text turns. History is text-only; prior image buffers are not retained. |
| `use_system_prompt_cache` | Request that a worker attempt system-prompt caching when supported. This is a capability request, not a guarantee that caching occurs. |

### Generation controls

| Field | Meaning |
| --- | --- |
| `max_generate_length` | Maximum number of output tokens requested. |
| `temperature` | Sampling temperature. |
| `top_p` | Nucleus-sampling probability. |
| `top_k` | Top-k sampling limit. |

These fields are part of the common request contract. Worker support is described below; the two current workers do **not** consume all generation fields identically.

---

## 2. Two independent mode axes

There are two different kinds of "mode" in a request and they should not be conflated.

### A. Media/sequence mode

`sequence_type` defines how the frame list is intended to be represented to the model:

| Value | Name | Intended meaning |
| ---: | --- | --- |
| `0` | `images` | One or more discrete image inputs. Order is preserved, but no native video semantics are requested. |
| `1` | `temporal_images` | An ordered temporal sequence whose timing matters. The current workers map this to their native-video path. |
| `2` | `video` | A video sequence. The current workers map this to their native-video path. |

### B. Instruction-delivery mode

Instruction delivery is independent of sequence type.

#### Inline delivery

The request contains a single prompt string and no native system/history roles.

Conceptually:

```text
current user message:
  media + complete rendered prompt
```

This is the default IPC format when neither `system_message` nor `history` is present.

#### Structured delivery

The request can carry:

```text
system:    <system_message>
user:      <historical user turn 1>
assistant: <historical model response 1>
...
user:      <current media + current prompt>
```

The `kSchemaFlagStructured` IPC bit indicates this representation. History entries are ordered oldest-first.

The standard C++ ROS node supports both instruction-delivery modes. In structured mode it keeps the system instruction and prior conversation turns out of the rendered current-user prompt so they are not duplicated.

---

## 3. IPC v3 wire request

The IPC protocol is a binary Unix-domain-socket protocol. Protocol v3 adds sequence semantics, FPS/timestamps, and temporal representation information to the earlier prompt/image contract.

### `RequestHeader`

| Header field | Meaning |
| --- | --- |
| `magic` | Protocol magic (`EVLM`). Rejects unrelated/corrupt traffic. |
| `version` | IPC schema version. Current version is `3`. |
| `request_id` | Monotonic client request identifier used to associate the response with the request. |
| `width`, `height` | Primary frame dimensions. |
| `step` | Primary BGR row stride. Packed transport requires `width * 3`. |
| `encoding` | IPC image encoding. Current workers require BGR8. |
| `image_bytes` | Byte size of the primary frame payload. |
| `prompt_bytes` | UTF-8 byte count of the current prompt/user text. |
| `max_generate_length` | Requested output-token cap. |
| `temperature` | Sampling temperature supplied by the client. |
| `top_p` | Nucleus-sampling parameter supplied by the client. |
| `top_k` | Top-k sampling parameter supplied by the client. |
| `schema_flags` | Bit mask describing structured delivery, extra images, timing metadata, and cache request. |
| `system_bytes` | UTF-8 byte count of the optional system message. |
| `history_count` | Number of prior `(user, assistant)` text turns. |
| `image_count` | Total frame count when multi-image is set; otherwise reserved as `0`. |
| `sequence_type` | `0=images`, `1=temporal_images`, `2=video`. |
| `fps` | Optional FPS value when the FPS flag is set. |
| `timestamp_count` | Number of `double` timestamps following the frame payload. Must equal total frame count when present. |
| `reserved` | Reserved for forward-compatible extension. |

### `schema_flags`

| Flag | Meaning |
| --- | --- |
| `kSchemaFlagStructured` | System message and/or prior history are carried using native message roles. |
| `kSchemaFlagSysCache` | Caller requests system-prompt caching if the worker/runtime supports it. |
| `kSchemaFlagMultiImage` | Two or more frames are carried. |
| `kSchemaFlagHasFps` | `RequestHeader.fps` is valid. |
| `kSchemaFlagHasFrameTimestamps` | A timestamp vector follows the image bytes. |

No bit is required for inline delivery; inline is the zero/default state for the instruction-delivery portion of the flags.

### Multi-frame metadata

The primary frame is described directly by `RequestHeader`. Every extra frame has a `PerImageHeader` containing:

```text
width
height
step
image_bytes
```

This allows individual frame dimensions to be transported in `images` mode. Native-video construction may impose stricter runtime requirements; the current Edge-LLM native-video implementation requires all frames to have identical width, height, and three RGB/BGR channels after decode.

### Wire payload order

After `RequestHeader`, the payload is sent in this exact order:

```text
RequestHeader

[PerImageHeader for frame 1]
[PerImageHeader for frame 2]
...
                    # only when multi-image is set

raw BGR8 bytes for frame 0
raw BGR8 bytes for frame 1
raw BGR8 bytes for frame 2
...

[double timestamp for frame 0]
[double timestamp for frame 1]
...
                    # only when timestamp flag is set

[system-message UTF-8 bytes]
                    # structured mode only; may be empty

current prompt UTF-8 bytes

[HistoryEntryHeader]
[historical user UTF-8 bytes]
[historical assistant UTF-8 bytes]
...
                    # structured mode only
```

Frame order on the wire is model-significant. The transport never sorts or reorders frames.

---

## 4. Sequence-mode semantics

### Mode: one image (`sequence_type=images`, one frame)

Logical request:

```text
frames:        [frame_0]
sequence_type: images
fps:           absent
timestamps:    absent
```

This is the normal mode used by the standard `edge_vlm_ros_node` today.

Both workers present one image to the VLM followed by the current text prompt.

### Mode: ordered multi-image (`sequence_type=images`, N frames)

Logical request:

```text
frames:        [frame_0, frame_1, ... frame_N-1]
sequence_type: images
fps:           absent
timestamps:    absent
```

Each frame is presented as an independent image media item in the current user message, in exactly the supplied order.

**Important:** order is preserved, but this mode does not request native-video temporal encoding. A model may infer relationships between ordered still images, but FPS and frame timing are not part of the model representation.

The repository records this distinction as ordered multi-image rather than native temporal/video inference.

### Mode: temporal images (`sequence_type=temporal_images`)

This protocol value declares that the sequence is temporal rather than merely a list of images.

In the current implementation, **both Edge-LLM and FlashRT route `temporal_images` through the same native-video construction used for `video`**. The value is still preserved at the request/provenance boundary so the caller's intended semantics are known.

A future runtime adapter could choose to distinguish `temporal_images` from `video`; clients should therefore continue to send the semantically correct sequence type rather than assuming the values are permanently interchangeable.

### Mode: video (`sequence_type=video`)

The ordered frame list is presented to the runtime as one native video media item rather than N independent image items.

Timing can be supplied by:

1. exact `frame_timestamps_sec`;
2. an explicit `fps`; or
3. worker fallback timing when neither is supplied.

For experiments where capture timing matters, exact frame timestamps are preferred because an average FPS alone cannot represent irregular intervals.

---

## 5. TensorRT Edge-LLM worker mapping

The standard worker executable is `edge_vlm_server`, implemented by `src/inference_worker.cpp`. It receives IPC v3, reconstructs an `InferenceRequest`, and calls `TensorRTEdgeLLMBackend::infer()`.

### Worker initialization

The worker loads once and remains resident across client connections:

```text
TensorRT Edge-LLM plugin
        +
LLM TensorRT engine
        +
multimodal TensorRT engine
        +
non-blocking CUDA stream
        -> persistent LLMInferenceRuntime
```

This means model/engine startup is not repeated for every inference request.

### Image preprocessing

IPC frames arrive as packed BGR8 `cv::Mat` images.

For every frame, the backend:

1. JPEG-encodes the OpenCV image in memory;
2. calls Edge-LLM `loadImageFromMemory()`;
3. obtains the runtime-native `ImageData` representation.

No temporary image file is required.

### Edge-LLM: `images` mode

For one frame:

```text
user message content:
  image <image>
  text  <prompt>

imageBuffers:
  ImageData(frame_0)
```

For N frames:

```text
user message content:
  image <image>
  image <image>
  ...
  text  <prompt>

imageBuffers:
  ImageData(frame_0)
  ImageData(frame_1)
  ...
```

Each frame is therefore a separate image content item and a separate image buffer.

Runtime provenance is reported as:

```text
ordered_multi_image_no_native_temporal_metadata
```

This provenance string is currently also used for the one-image path.

### Edge-LLM: `temporal_images` and `video`

Both values select `use_native_video=true`.

The backend:

1. decodes all submitted frames;
2. requires identical frame dimensions/channels;
3. stacks them into one CPU tensor with shape:

```text
[T, H, W, 3]
```

4. creates one Edge-LLM `ImageData` object;
5. sets `ImageData.isVideo = true`;
6. assigns an effective FPS;
7. attaches exact timestamps when they were provided by the caller;
8. places one `video` content item in the current user message;
9. places one video `ImageData` in `imageBuffers`.

Conceptually:

```text
user message content:
  video <video>
  text  <prompt>

imageBuffers:
  ImageData(
    tensor=[T,H,W,3],
    isVideo=true,
    fps=<effective fps>,
    timestamps=<optional exact timestamps>
  )
```

Runtime provenance is reported as:

```text
native_qwen3vl_video_imagedata_mrope_timestamps
```

#### Edge-LLM effective FPS

The current Edge-LLM adapter chooses effective FPS in this order:

1. explicit request `fps`, if present;
2. `(frame_count - 1) / (last_timestamp - first_timestamp)` when timestamps are present;
3. `1.0 fps` fallback.

When exact timestamps are present, they are also attached to `ImageData`; the effective FPS does not replace the timestamp vector.

### Edge-LLM message roles

Structured requests map directly to native runtime messages:

```text
system message               -> role=system, text only
history user text            -> role=user
history assistant text       -> role=assistant
current media + prompt       -> role=user
```

Historical entries contain text only. Previous images are not replayed with conversation history.

### Edge-LLM generation invocation

The backend creates one `LLMGenerationRequest` and currently applies:

```text
temperature       <- request.temperature
topP              <- request.top_p
topK              <- request.top_k
maxGenerateLength <- request.max_generate_length
applyChatTemplate = true
addGenerationPrompt = true
```

Inference is then invoked through:

```text
LLMInferenceRuntime::handleRequest(...)
```

### Edge-LLM system-prompt cache status

The IPC protocol transports `use_system_prompt_cache` through `kSchemaFlagSysCache`, but the current TensorRT Edge-LLM backend does **not yet enable a runtime cache API**. The code intentionally leaves this as an opt-in capability placeholder pending validation of the exact API on the pinned runtime.

Therefore:

```text
cache flag transported: yes
cache flag currently changes runtime behavior: no
```

---

## 6. FlashRT Cosmos3 worker mapping

The FlashRT worker is `flashrt_temporal/flashrt_ipc_worker.py`.

It reads the same IPC v3 header, BGR frame payload, timing metadata, prompt, optional system message, and optional history used by the C++ worker.

IPC BGR8 frames are converted to PIL RGB images before Cosmos3 preprocessing.

### FlashRT message construction

Structured message roles are reconstructed as:

```text
system_message     -> {role: system, content: text}
history user       -> {role: user, content: text}
history assistant  -> {role: assistant, content: text}
current media/text -> {role: user, content: [...]}
```

The worker explicitly renders/uses the chat template with:

```text
enable_thinking = false
```

for the current Cosmos3 path.

### FlashRT: `images` mode

Every frame is added as a separate image item:

```text
current user content:
  {type: image, image: frame_0}
  {type: image, image: frame_1}
  ...
  {type: text,  text: prompt}
```

The processor returns image tensors such as:

```text
pixel_values
image_grid_thw
```

FlashRT invocation then uses:

```text
is_video = false
```

Runtime provenance distinguishes:

```text
flashrt_single_image
flashrt_ordered_multi_image
```

based on frame count.

### FlashRT: `temporal_images` and `video`

Both values are treated as native video.

The current user message contains one video media object:

```text
{type: video, video: [frame_0, frame_1, ...]}
{type: text,  text: prompt}
```

The worker stacks the RGB frames into an array representing the submitted temporal sequence and invokes the Cosmos3 processor with:

```text
videos=<stacked frames>
video_metadata=<derived metadata>
do_sample_frames=False
return_tensors="pt"
```

`do_sample_frames=False` is important: the frames have already been selected by the upstream ROS/experiment sampler. The Cosmos processor is not asked to choose a second subset and silently change the experiment.

The processed request is passed to FlashRT using:

```text
pixel_values = pixel_values_videos
grid_thw     = video_grid_thw
is_video     = true
```

Runtime provenance is:

```text
flashrt_cosmos3_native_video
```

### FlashRT timestamp mapping

When exact timestamps are supplied, the worker preserves their relative placement using a synthetic 1000 Hz metadata timebase.

Example input timestamps:

```text
[100.000, 100.267, 100.534, 100.868]
```

become relative positions approximately:

```text
frames_indices = [0, 267, 534, 868]
metadata fps   = 1000.0
```

The important quantity is the frame index relative to that 1000 Hz timebase, not a claim that the source camera actually operated at 1000 fps. This representation lets the processor recover the irregular relative timing of frames that were already sampled upstream.

Metadata includes:

```text
total_num_frames
fps
duration
frames_indices
video_backend
```

When exact timestamps are absent, the FlashRT worker instead uses sequential frame indices and an effective FPS.

#### FlashRT effective FPS

For diagnostics/fallback timing, FlashRT chooses effective FPS in this order:

1. explicit request `fps`, if present;
2. reciprocal of the median inter-frame timestamp interval when timestamps are present;
3. `1.0 fps` fallback.

This differs slightly from the current Edge-LLM adapter, which derives timestamp-only effective FPS from the total first-to-last span.

### FlashRT generation invocation

The worker validates that:

```text
request.max_generate_length > 0
request.max_generate_length <= worker engine_max_new_tokens
```

It then invokes:

```text
CosmosReasonerThor.generate(
    input_ids,
    max_new_tokens=request.max_generate_length,
    ignore_eos=False,
    ...media tensors...
)
```

#### Current sampling-control limitation

Although IPC v3 carries `temperature`, `top_p`, and `top_k`, the current FlashRT worker does **not** pass those fields into `CosmosReasonerThor.generate()`.

Therefore, today:

| Generation field | Edge-LLM worker | FlashRT worker |
| --- | --- | --- |
| `max_generate_length` | used | used |
| `temperature` | used | transported/parsed but not used by worker |
| `top_p` | used | transported/parsed but not used by worker |
| `top_k` | used | transported/parsed but not used by worker |

This is a runtime-adapter difference, not an IPC limitation.

### FlashRT system-prompt cache status

The FlashRT worker can reconstruct structured system/history roles, but it does not currently implement the IPC system-prompt-cache request. The cache bit therefore has no runtime effect on this worker.

---

## 7. Current client behavior

Worker capability and current client behavior are deliberately separated. A worker may support a protocol mode that a particular ROS client does not currently expose.

### Standard C++ ROS node

`edge_vlm_ros_node` currently constructs one-frame `InferenceRequest` objects:

```text
image              = current sampled BGR frame
extra_images       = empty
sequence_type      = images (default)
fps                = absent
frame_timestamps   = absent
prompt             = rendered current task prompt
max_generate_length / temperature / top_p / top_k = configured ROS parameters
```

It can use either inline or structured instruction delivery.

In structured mode it additionally supplies:

```text
system_message
history[]
use_system_prompt_cache request flag
```

Multi-frame and native-video support exists below this node at the request/IPC/worker level and is exercised by benchmark/experiment clients, but the standard live C++ node does not currently build rolling frame windows.

### FlashRT temporal ROS node

`flashrt_temporal/temporal_ros_node.py` currently builds rolling frame windows and submits them to the FlashRT worker as:

```text
frames             = current contiguous temporal window
timestamps         = exact ROS message timestamps
sequence_type      = video
fps                = absent
prompt             = configured temporal prompt
max_generate_length= configured parameter
instruction mode   = inline
```

It currently sets these IPC generation fields:

```text
temperature = 0.0
top_p       = 1.0
top_k       = 1
```

but, as noted above, the FlashRT worker currently does not consume those three sampling fields.

The temporal node's `IpcClient` currently exposes `images` and `video`; the FlashRT **worker** also understands the protocol value `temporal_images`, even though this particular client does not expose it as a selectable mode today.

The temporal ROS node always sends exact timestamps and does not send an FPS value.

---

## 8. Mode matrix

| Request form | IPC representation | Edge-LLM worker | FlashRT worker |
| --- | --- | --- | --- |
| Single still image | 1 BGR frame, `sequence_type=images` | one image content item / `ImageData` | one image content item / image tensors |
| Ordered still images | N BGR frames, `sequence_type=images` | N image content items; no native temporal metadata | N image content items; `is_video=false` |
| Temporal images | N frames, `sequence_type=temporal_images` | native video `ImageData` | native Cosmos3 video path |
| Video | N frames, `sequence_type=video` | native video `ImageData` | native Cosmos3 video path |
| Inline instruction | prompt only | current user text | current user text |
| Structured instruction | system + history + current prompt | native system/user/assistant roles | system/user/assistant chat messages |
| Exact timestamps | N doubles, strictly increasing | attached to native video `ImageData` | mapped into video metadata frame positions |
| FPS only | header FPS | assigned to native video | used in video metadata |
| Sampling controls | header generation fields | max tokens + temperature/top-p/top-k used | only max tokens currently used |
| System-prompt cache request | schema flag | transported but runtime activation not yet implemented | not implemented |

---

## 9. Response contract

Both workers respond using the same IPC `ResponseHeader` followed by variable-length UTF-8 payloads.

| Response field | Meaning |
| --- | --- |
| `request_id` | Must match the submitted request. |
| `success` | Worker/model inference success flag. |
| `text_bytes` | Generated response byte count. |
| `error_bytes` | Error-message byte count. |
| `temporal_encoding_bytes` | Byte count of the runtime-representation provenance string. |
| `temporal_fallback_used` | Indicates whether a requested temporal representation degraded to another representation. |
| `inference_seconds` | Worker-measured inference duration. |

The variable payload then contains:

```text
generated text
error text
runtime temporal-encoding string
```

The ROS side normalizes these values into `edge_vlm_ros/msg/VlmResult`.

The temporal-encoding string is important experimental provenance. `sequence_type=video` tells us what the caller requested; `runtime_temporal_encoding` tells us what the worker actually used.

---

## 10. Practical interpretation rules

When reading a result or benchmark artifact, use these rules:

1. **Frame count alone does not imply temporal video inference.** Eight frames with `sequence_type=images` are ordered still images; eight frames with `sequence_type=video` use the native-video path.
2. **Order and timing are separate.** Ordered images preserve sequence order without supplying physical timing. Native video can additionally carry FPS or exact timestamps.
3. **Exact timestamps are part of the input semantics.** Replacing irregular timestamps with an average FPS changes the represented sequence.
4. **`temporal_images` and `video` currently share a worker path, but remain separate protocol values.** Do not erase the distinction from stored experiment metadata.
5. **Instruction delivery is orthogonal to media mode.** Any supported sequence type can in principle be paired with inline or structured text delivery.
6. **Requested controls are not proof that a worker used them.** In particular, FlashRT currently ignores IPC temperature/top-p/top-k, and neither current worker activates system-prompt caching.
7. **Record both requested and actual representation.** Use `sequence_type` plus `runtime_temporal_encoding` when comparing experiments.

## Source-of-truth implementation files

The contract described here is implemented in:

- `include/edge_vlm_ros/inference_backend.hpp` — logical C++ request/response types and temporal validation;
- `include/edge_vlm_ros/ipc_protocol.hpp` — IPC v3 binary structures and flags;
- `src/ipc_inference_backend.cpp` — C++ IPC request serialization;
- `src/inference_worker.cpp` — C++ worker IPC deserialization and request reconstruction;
- `src/tensorrt_edge_llm_backend.cpp` — TensorRT Edge-LLM runtime mapping;
- `flashrt_temporal/ipc_protocol.py` — Python mirror of IPC v3;
- `flashrt_temporal/flashrt_ipc_worker.py` — FlashRT/Cosmos3 runtime mapping;
- `flashrt_temporal/temporal_ros_node.py` — current rolling-window FlashRT ROS client.

If the implementation and this document diverge, treat that as a documentation bug and update both together.