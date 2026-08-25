# Temporal VLM architecture

Status: **target design with validated runtime elements**  
Tracking: #8, #72 / PR #74, #75

## Purpose

The temporal VLM path answers a different question from single-frame image reasoning:

> Given a bounded recent visual history, what changed, in what direction, and when did it become evident?

Temporal context construction and temporal model representation are therefore explicit subsystems rather than incidental prompt details.

```text
camera / rosbag
    |
    v
bounded rolling window + sampling/backpressure
    |
    v
explicit sequence representation + timestamps
    |
    v
model/runtime adapter
    |
    v
structured temporal observation
```

## Architectural boundaries

### Acquisition and sampling

The ROS side owns sensor subscription, sampling interval, continuity checks, and backpressure. These policies should not be embedded inside a particular model runtime.

A live temporal window must remain bounded. If inference is slower than the camera stream, the scheduler keeps the newest useful pending window rather than building an unbounded queue.

### Continuity

A temporal sequence is only meaningful if the source interval is continuous enough for the task.

The current ROS temporal node resets its window when:

- timestamps move backward;
- a forward gap exceeds the configured maximum;
- a replay loop crosses back to an earlier source timestamp.

This prevents unrelated bursts from being silently presented to the model as one continuous clip.

### Representation

The runtime contract distinguishes at least:

- single image;
- ordered independent images;
- temporal image sequence;
- native video.

Those are different model-input conditions. A benchmark should never relabel ordered images as video without verifying that the processor/runtime path actually changes.

### Timing

Frame timing is part of temporal semantics. Exact capture timestamps are preferred when available.

For pre-sampled frames, an average FPS is not automatically equivalent to the source timing. Irregularly spaced frames should either retain exact timestamps, be explicitly resampled to a uniform timeline, or be rejected for experiments that require controlled timing.

### Runtime adapter

The model adapter converts the generic IPC request into the processor-specific representation expected by the selected VLM. It is responsible for recording the actual runtime temporal encoding in provenance.

For the FlashRT/Cosmos3 path, native video requests are delivered as one video tensor with explicit video metadata derived from the supplied timestamps.

## Scheduling model

The live pipeline uses latest-only scheduling:

```text
camera frames
    -> sampled frame/window
    -> [active inference]
       while busy: replace pending input with newest candidate
    -> next inference uses newest pending input
```

This is intentional. The pipeline is designed for bounded semantic observation, not for guaranteeing inference on every camera frame.

## Output contract

Temporal model output should be normalized into fields that can be scored mechanically. Useful concepts include:

- whether change was detected;
- object or scene element involved;
- direction/trend;
- start/end state;
- evidence interval;
- camera motion;
- confidence or uncertainty;
- runtime/model provenance.

Free-form summaries can remain useful for humans but should not be the sole evaluation signal because they are harder to score and may contradict structured fields.

## Provenance requirements

Every controlled temporal result should record:

- ordered source frame identifiers;
- exact timestamps or explicit resampling policy;
- frame count and window span;
- sequence type;
- input representation;
- runtime temporal encoding;
- model and engine identity;
- quantization/profile;
- prompt version and prompt hash when available;
- generation limit and sampling configuration;
- inference/client latency.

Without those fields, two apparently identical frame sequences can represent materially different experiments.

## Controlled temporal evaluation

The minimum chronology suite is:

1. **chronological** — original frame order;
2. **reversed** — exact same images reversed, with the same monotonic timestamp schedule;
3. **shuffled** — deterministic non-chronological order;
4. **repeated static** — one source frame repeated across the full native-video schedule;
5. **single terminal frame** — diagnostic for action-state inference from appearance.

Additional useful controls include dropped frames, sparse sampling, different window lengths, camera-only motion, and multiple independent moving objects.

The controlled Cosmos3 result from 2026-08-25 demonstrated chronology sensitivity on one motion-rich sequence: reversing the exact same eight images reversed the reported lateral motion direction. The shuffled control still produced a plausible motion story, so general temporal-coherence reliability remains open.

See [`../temporal-chronology-results.md`](../temporal-chronology-results.md).

## Model comparison strategy

Once a native-video path works for one model, model-specific prompt tuning should pause long enough to compare other video-capable models on identical saved captures.

A shared benchmark should score:

- forward direction/change accuracy;
- reverse consistency;
- static false-change rate;
- shuffled-sequence rejection or uncertainty;
- camera-motion accuracy;
- structured-output compliance;
- contradiction rate;
- inference latency;
- memory/engine footprint.

This keeps the repository focused on a reusable ROS 2 VLM pipeline rather than optimizing around one model's quirks.

## Distillation and specialization

Teacher/student training must preserve the temporal representation used to create labels. If the teacher saw native video with exact timing, training/evaluation should not silently convert those examples into independent still images and claim equivalent semantics.

Specialization is most useful after comparative evaluation identifies a model/representation with a promising baseline quality/latency tradeoff.

## Design invariants

1. Temporal windows are bounded.
2. Source discontinuities reset the window.
3. Sequence representation is explicit.
4. Exact timing is preserved or deliberately transformed and recorded.
5. Scheduling/backpressure is separate from model preprocessing.
6. Runtime temporal encoding is recorded in provenance.
7. Controlled model comparisons use identical saved evidence.
8. Structured fields are preferred for scoring over free-form narrative text.
