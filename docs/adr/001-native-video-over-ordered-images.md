# ADR 001: Prefer native video semantics for temporal experiments

- Status: Accepted
- Date: 2026-08-24
- Related: #72 / PR #74, #75

## Context

The project initially established multi-frame latency by sending multiple ordered image content items to Cosmos-Reason2. That proved that additional visual context can be inexpensive relative to generation latency, but it did **not** prove native temporal/video reasoning.

For Qwen3-VL/Cosmos-Reason2, the runtime has a distinct video representation with temporal grouping, timestamp handling, and multimodal positional semantics. A list of independent images and one native video object may contain identical source pixels while presenting materially different temporal information to the model.

Live ROS inputs naturally arrive as individual frames, so requiring an encoded MP4/container merely to obtain video semantics would add unnecessary I/O and encoding work.

## Decision

When evaluating short-horizon temporal reasoning and the runtime supports it, use the model's **native video representation** rather than treating ordered independent images as the primary temporal path.

For live/pre-sampled frames, native video may be constructed directly from a frame stack rather than from an encoded video file.

Independent ordered images remain a supported baseline/control.

Every result must report the effective runtime representation so a temporal request cannot silently fall back to ordered images without changing experiment provenance.

## Consequences

### Positive

- Experiments exercise the model's intended temporal semantics.
- Timing metadata can be represented explicitly rather than inferred from image order alone.
- Live ROS frames do not need to be written to temporary video files.
- Ordered-image and native-video behavior can be compared as controlled variants.
- Provenance makes representation changes visible.

### Negative / cost

- Runtime and IPC schemas must carry temporal metadata.
- Model-specific temporal behavior must be understood and tested.
- Exact timestamp handling differs between entry paths and processor/runtime versions.
- Native-video regressions require hardware validation in addition to CPU-only tests.

## Alternatives considered

### Ordered independent images only

Rejected as the primary temporal representation because ordering alone does not establish native video timing semantics.

### Encode each window to MP4 before inference

Rejected for live/pre-sampled frame windows because it adds needless encoding, filesystem/memory movement, latency, and another failure mode.

### Render timestamps into image pixels

Useful as an experimental control, but rejected as the primary representation because it changes image content and does not substitute for native temporal metadata/position handling.

## Verification

PR #74 smoke testing on Jetson AGX Thor demonstrated F4 and F8 native-video inference with:

```text
runtime_temporal_encoding = native_qwen3vl_video_imagedata_mrope_timestamps
temporal_fallback_used = false
```

That validates feasibility of the chosen runtime path; task-level quality still requires controlled evaluation.
