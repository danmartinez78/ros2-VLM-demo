# Temporal VLM architecture

Status: **target design with validated runtime elements**  
Tracking: #8, #72 / PR #74, #75

## Purpose

The temporal VLM path exists to answer a different question from single-frame image reasoning:

> Given a bounded recent visual history, what changed, in what direction, and when did it become relevant?

The architecture therefore treats **temporal context construction** and **temporal model representation** as explicit subsystems rather than as incidental details of a prompt.

The intended end-to-end path is:

```text
camera / rosbag
    |
    v
bounded rolling window + sampling/backpressure
    |
    v
explicit sequence request
(images | temporal_images | video)
    |
    v
versioned IPC contract
    |
    v
persistent Edge-LLM runtime
    |
    v
Qwen3-VL / Cosmos-Reason2 temporal processing
    |
    v
temporal assessment
    |
    +--> task-level evaluation
    +--> ODD-axis estimator / downstream observation consumer
```

The design goal is not to maximize the number of frames presented to the model. It is to maximize **useful recent temporal evidence per unit latency and compute**, while making the representation used for every experiment observable and reproducible.

## Current and target boundaries

### Validated/current

The repository uses a persistent ROS-free GPU worker separated from the ROS process. That process boundary is documented in [`../architecture.md`](../architecture.md) and is required on the validated Thor stack.

The multi-frame benchmark work established that Cosmos-Reason2 can consume ordered multi-image inputs with relatively modest latency growth as frame count increases.

PR #74, now merged into `main`, added the explicit temporal sequence contract and native Qwen3-VL/Cosmos-Reason2 video path. On Thor, smoke testing demonstrated:

- F4 native video requests succeed without temporal fallback;
- F8 native video requests succeed without temporal fallback;
- the runtime reports `native_qwen3vl_video_imagedata_mrope_timestamps` as the effective encoding.

### Target design

Issue #8 owns live rolling-window construction and backpressure. The runtime should receive an already-selected bounded sequence; it should not grow an unbounded camera queue or make hidden scheduling choices.

The downstream ODD architecture is described separately in [`odd-observation-system.md`](odd-observation-system.md).

## Three different rates

Temporal experiments must distinguish three rates that are often incorrectly collapsed into a single "FPS" value.

### Camera FPS

The physical or replayed sensor production rate.

Example:

```text
camera_fps = 30 Hz
```

This describes how frequently frames become available. It does **not** imply that 30 frames should be sent to the VLM every second.

### Context FPS

The effective sampling density represented inside one VLM request.

For `N` frames covering duration `D`:

```text
context_fps ~= N / D
```

A request may therefore contain eight frames representing one recent second even if the source camera runs at 30 Hz.

### Inference FPS

The rate at which complete VLM requests can be serialized through the current worker/model configuration.

If one request requires 600 ms end-to-end, serialized inference throughput is roughly:

```text
1 / 0.6 s ~= 1.67 requests/s
```

This is independent of the context density inside each request.

A useful operating point may therefore look like:

```text
camera:       30 Hz
context:       8 Hz inside each rolling window
VLM requests: ~1.5 Hz
```

The rolling-window scheduler must keep the context **fresh** even when inference is much slower than the source camera.

## Sequence representation contract

The model input representation is part of the experiment definition. Identical JPEG bytes delivered through different representations are not assumed to be equivalent examples.

### `images`

Ordered independent image content items.

```text
frame 0 -> image item
frame 1 -> image item
frame 2 -> image item
...
```

Properties:

- preserves order in the request;
- does not, by itself, provide native video timing semantics;
- useful as a baseline and compatibility path;
- runtime representation reports ordered multi-image semantics explicitly.

Current runtime provenance label:

```text
ordered_multi_image_no_native_temporal_metadata
```

### `temporal_images`

A sequence of already-decoded/sampled image frames intentionally delivered through the model's native video representation rather than as independent images.

This mode exists because a live ROS camera naturally produces frames, not MP4 files. Native video semantics should not require creating a temporary encoded video container.

Conceptually:

```text
JPEG/RGB frame list
    -> decode
    -> stack [T,H,W,C]
    -> one ImageData
       isVideo = true
       fps = ...
       timestamps = ...
    -> one video content item
```

### `video`

A sequence explicitly declared as video. Depending on the entry path, this may be built from a frame list or an encoded media source, but the effective runtime representation must still be reported.

For the repository's pre-sampled frame use case, `video` and `temporal_images` can converge on the same Edge-LLM `ImageData` representation while retaining distinct request intent/provenance.

## Native Qwen3-VL / Cosmos-Reason2 temporal representation

At the pinned TensorRT Edge-LLM revision used by the project, `ImageData` can represent a frame stack:

```text
buffer:     [T,H,W,C] uint8
frames:     T
fps:        source/effective FPS
isVideo:    true
timestamps: optional explicit frame timestamps
```

The Qwen3-VL/Cosmos-Reason2 runner uses the video path to build temporal visual groups and model timing information. The relevant temporal behavior includes:

- temporal patch grouping rather than treating every frame as an unrelated image;
- video-specific chat-template placeholders;
- timestamp text associated with temporal groups;
- interleaved multimodal rotary position handling (MRoPE).

The implementation must not infer success merely because multiple frames were accepted. The response reports the **effective runtime temporal encoding**.

## No-silent-fallback invariant

A temporal request must never be reported as native temporal/video processing if the runtime actually delivered independent images.

Every temporal inference result should preserve at least:

```text
requested_sequence_type
effective/runtime_temporal_encoding
temporal_fallback_used
fps or timestamp policy
frame count
```

The experiment harness must treat a fallback as a representation change, not as an implementation detail.

For example:

```text
requested_sequence_type = video
runtime_temporal_encoding = ordered_multi_image_no_native_temporal_metadata
temporal_fallback_used = true
```

is a different experiment from:

```text
requested_sequence_type = video
runtime_temporal_encoding = native_qwen3vl_video_imagedata_mrope_timestamps
temporal_fallback_used = false
```

## Explicit timestamps versus FPS

Uniformly sampled sequences can be represented by a frame rate:

```text
t_i = i / fps
```

Irregularly sampled sequences require explicit timestamps if exact timing matters.

Do not approximate irregular timing with a single average FPS in experiments intended to evaluate temporal reasoning. Either:

1. preserve exact timestamps through a supported processor/runtime path;
2. deterministically resample to a uniform sequence and record that transformation; or
3. reject the sample for that experiment.

The distillation pipeline applies the same principle; see [`../distillation-pipeline-design.md`](../distillation-pipeline-design.md).

## Rolling-window ownership (#8)

Issue #8 owns **which frames are selected** for each inference request.

The scheduler should provide:

- timestamp-ordered recent-frame storage;
- bounded capacity by frame count and/or duration;
- deterministic sampling/downsampling;
- freshness guarantees;
- explicit handling when inference is slower than incoming frames;
- no unbounded queue;
- observability for dropped/replaced frames and effective context age.

A representative flow is:

```text
30 Hz camera
   |
   v
recent frame ring/buffer
   |
   +-- old frames age out
   +-- newest frames replace stale candidates
   |
request trigger (~1.5 Hz)
   |
   v
sample recent 8-frame context
   |
   v
#74 temporal representation + IPC
```

### What #8 does not own

#8 should not:

- decide whether Edge-LLM uses native video or independent image content;
- construct model-specific MRoPE metadata;
- hide runtime fallback;
- own TensorRT engine/profile selection;
- accumulate every source frame while inference is busy.

Those responsibilities belong to the temporal request/runtime layer or model-management layer.

## Backpressure and freshness

For live observation, freshness is generally more valuable than processing every frame.

If inference is serialized and a request is in flight:

```text
DO:
  keep a bounded recent context
  replace stale pending candidates
  build the next request from the newest eligible window

DO NOT:
  append every source frame to an unbounded FIFO
  process old windows long after the scene has changed
```

Useful scheduler observability includes:

- newest source timestamp;
- oldest timestamp in selected context;
- context duration;
- selected frame count;
- context FPS;
- request trigger timestamp;
- frame age when inference begins;
- source frames dropped/replaced since the prior request.

## Latency interpretation

Measured latency must be tied to representation and output workload.

A useful record includes:

```text
model / engine identity
frame count
sequence type
runtime temporal encoding
context FPS / timestamp policy
input resolution
output token budget
actual output tokens when available
vision time
prefill time
generation time
end-to-end latency
```

Do not infer temporal-representation efficiency from a single run when generated response lengths differ materially.

The existing Cosmos-Reason2 measurements indicate that generation often dominates total latency, while additional frames primarily increase vision and prefill cost. That makes short structured outputs and bounded temporal context important architecture levers.

## Quality evaluation

Temporal reasoning quality must be tested independently of latency.

Core controlled tasks include:

- appears / disappears;
- approaches / recedes;
- rain/fog increases or decreases;
- road becomes blocked / unblocked;
- construction-zone transition;
- static/no-change sequences.

Each task should include controls such as:

- chronological sequence;
- reversed sequence;
- shuffled sequence;
- duplicated frame;
- terminal-frame-only input.

Metrics should emphasize semantic correctness rather than brittle references to frame numbers:

- change-detection accuracy;
- direction/trend accuracy;
- event-time bucket accuracy;
- static false-positive rate;
- schema adherence;
- hallucinated frame/entity references;
- confidence/calibration when the target schema supports it;
- latency under the same representation.

## Provenance requirement

Every temporal experiment should be reconstructable from its artifacts.

At minimum record:

```text
repository commit
model and engine identity
sequence_type
runtime_temporal_encoding
frame paths or source sequence identity
frame timestamps or timestamp policy
effective FPS
frame count
sampling policy
rendered timestamp control (if any)
prompt profile/version
generation settings
```

This is necessary because the same source frames can represent different experiments when timing, ordering, or runtime encoding changes.

## Relationship to distillation

The student should be trained on the same temporal semantics it is expected to use at inference time.

Therefore:

```text
teacher representation
        ~=
training processor representation
        ~=
student evaluation representation
        ~=
target runtime representation
```

The equality is semantic rather than byte-for-byte, but timing/order information must not silently change. ADR 003 records this decision.

## Deferred / experimental memory approaches

Native short video context is the baseline temporal representation. Longer-horizon memory remains experimental and should be evaluated as separate mechanisms, including:

- recurrent text world-state/history;
- detector + tracker context upstream of the VLM;
- learned sequence memory such as a Mamba-style layer;
- knowledge-graph state with deterministic retrieval into the prompt.

These mechanisms may complement a short native-video window; they should not be conflated with the basic frame-window representation problem.

## Non-goals

This architecture does not claim that:

- a VLM is the best estimator for every ODD axis;
- more frames always improve quality;
- native video alone solves long-horizon memory;
- VLM output should directly trigger a safety maneuver;
- timestamp text emitted by a model is itself proof that internal timing is correct.

Those questions require task-specific evaluation and, where relevant, separate safety architecture.
