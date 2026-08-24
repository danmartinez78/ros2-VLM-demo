# ADR 002: Keep bounded-window scheduling separate from VLM runtime representation

- Status: Accepted
- Date: 2026-08-24
- Related: #8, #72 / PR #74, #75

## Context

A live camera can produce frames much faster than a serialized VLM can process requests. The system therefore needs a policy for retaining recent frames, selecting a bounded context, dropping stale work, and constructing the next request.

Separately, the inference runtime needs to know **how the already-selected frames are represented to the model**: independent images, native temporal images, or video with timing metadata.

Combining these responsibilities in one model/runtime component would couple camera-rate behavior, backpressure, context freshness, IPC semantics, and model-specific representation. It would also make it difficult to test scheduling independently from the VLM.

## Decision

Keep two explicit layers:

```text
camera / rosbag
    |
    v
bounded rolling-window scheduler (#8)
    |
    v
selected ordered frame set + timing
    |
    v
temporal representation / IPC / VLM runtime (#72 / PR #74)
```

Issue #8 owns:

- timestamp-ordered recent-frame storage;
- bounded capacity by duration and/or count;
- deterministic sampling/downsampling;
- freshness and backpressure;
- replacement/drop policy when inference is busy;
- observability of selected context and dropped source frames.

The temporal runtime layer owns:

- `images | temporal_images | video` request semantics;
- serialization of FPS/timestamps;
- native model video construction;
- effective runtime-representation reporting;
- no-silent-fallback behavior.

## Consequences

### Positive

- Scheduling can be unit-tested without loading a VLM.
- The same selected window can be replayed through different model representations for controlled experiments.
- Model/runtime changes do not require rewriting camera backpressure policy.
- Alternative schedulers can be evaluated without changing IPC semantics.
- Unbounded queues are explicitly prevented before inference.

### Negative / cost

- The interface between the scheduler and runtime must preserve ordering/timing correctly.
- There are two configuration surfaces that must be reported together for experiment provenance.
- Some optimizations that span both layers may require coordinated changes.

## Alternatives considered

### Let the VLM worker own the camera queue

Rejected because the GPU/runtime process should not accumulate source frames or own ROS/source-rate policy. It also weakens the existing process-isolation boundary.

### FIFO every sampled frame until processed

Rejected because inference throughput can be much lower than source throughput, causing stale contexts and unbounded latency/memory growth.

### Build a fresh window only after the prior inference completes

Possible as a simple implementation, but still belongs in the scheduling layer. It may also miss useful deterministic trigger/freshness behavior if treated as an incidental runtime detail.

## Operational invariant

When inference falls behind, the system should prefer **fresh bounded context** over complete processing of historical windows.
