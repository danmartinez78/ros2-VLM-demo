# Shared vLLM serving for modular ROS VLM pipelines

Status: **target design / experiment proposal**  
Related: #9, #73, #80, #82, #84

## Purpose

The ROS pipeline should not require one foundation-model process per perception function. A serving-oriented backend such as vLLM creates a useful alternative architecture: multiple logically independent ROS callers can share one loaded VLM while keeping task decomposition modular.

The central design question is:

> Can a shared OpenAI-compatible VLM server provide acceptable single-request latency while improving aggregate throughput, model-memory efficiency, and multi-caller behavior compared with a dedicated IPC worker?

This document defines the intended boundary and the experiments needed to answer that question. It does **not** replace the existing IPC path.

## Design principle

The stable abstraction is the ROS inference/result contract. Runtime adapters remain replaceable.

```text
camera / rosbag / perception context
            |
            v
sampling / temporal policy
            |
            v
logical inference request
            |
      +-----+-----+
      |           |
      v           v
  IPC backend   vLLM backend
      |           |
 Unix socket   OpenAI-compatible API
      |           |
 FlashRT /     shared model server
 Edge-LLM          |
      |           |
      +-----+-----+
            |
            v
       normalized VlmResult
```

The IPC backend remains appropriate when the application needs tight control over native frame/timestamp representation, minimum local overhead, or a runtime that is not exposed through a serving API.

The vLLM backend is attractive when model sharing, concurrent callers, standardized request/response semantics, remote serving, and easier model substitution are valuable.

## Keep backend-specific nodes separate

Prefer separate ROS backend nodes rather than a single node with a large runtime switch.

Example:

```text
vlm_ipc_node
    ROS request -> versioned IPC -> local accelerator worker

vlm_vllm_node
    ROS request -> OpenAI-compatible HTTP -> vLLM server
```

Both should normalize backend-specific responses into the same downstream result contract.

This keeps:

- Unix-socket/shared-memory details out of the vLLM path;
- HTTP/media/API details out of the IPC path;
- runtime lifecycle concerns localized;
- benchmarking and downstream consumers backend-neutral.

## Shared-server topology

A primary motivation for vLLM is supporting multiple perception functions without loading multiple copies of the same foundation model.

```text
                       one loaded VLM
                            |
                       vLLM server
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
   weather observer   temporal observer   scene/ODD observer
       slow rate          medium rate          independent rate
```

Each caller can remain a small, task-focused ROS component with its own:

- prompt/task profile;
- input/context selection;
- temporal window;
- sampling density;
- output-token budget;
- requested cadence;
- priority class if the backend supports and the experiment justifies it.

This avoids forcing weather, scene semantics, temporal motion, terrain, construction-zone reasoning, and other concerns into one monolithic VLM node merely to share model memory.

## Why a serving runtime may help under concurrency

A dedicated worker commonly processes one request at a time unless the repository implements its own batching and arbitration. Multiple callers therefore tend toward either:

```text
one model per caller
```

which wastes model memory, or:

```text
many callers -> one serial worker queue
```

which can create poor tail latency and requires custom scheduling logic.

A serving-oriented runtime is designed to own that scheduling problem. The expected advantages to evaluate are:

- one resident model shared by many callers;
- continuous/dynamic batching of compatible work;
- better GPU utilization during decode;
- managed KV-cache and request lifecycle;
- asynchronous request handling;
- standardized usage/timing metadata;
- simpler local or remote client integration.

These are **hypotheses to benchmark on Thor**, not assumptions that concurrency is free. Concurrent requests still share finite GPU compute and may increase individual request latency.

## Throughput versus latency

The correct comparison is not only single-request latency.

For concurrency levels `1, 2, 4, 8`, measure at least:

- request latency p50 / p95;
- time to first token when available;
- completion/decode tokens per second;
- aggregate requests per second;
- aggregate generated tokens per second;
- GPU and unified-memory use;
- failure/timeout rate;
- queue/wait time;
- result staleness relative to source timestamps.

A shared backend can be the better system even if concurrency=1 latency is slightly worse, provided aggregate throughput and tail behavior under realistic multi-caller load improve enough to justify it.

## Temporal policies remain caller-specific

A shared model server should **not** imply one global temporal sampling policy.

Different functions may submit different observation regimes to the same model, for example:

```text
weather / slow scene state
    5 s window @ 2-3 fps
    richer output

moderate temporal observer
    3 s window @ 5 fps
    moderate output

fast dynamic-scene observer
    1-2 s window @ 8-10 fps
    compact structured output
```

The temporal scheduler remains outside the model runtime. This preserves the existing architecture invariant that acquisition, window management, continuity, and backpressure are separate from model preprocessing.

Longer term, an adaptive scheduler may tune:

- window duration;
- temporal sample rate;
- stride / overlap;
- output-token budget;
- possibly image resolution;

based on ego dynamics, scene dynamics, recent inference latency, and task freshness requirements.

A shared serving backend is complementary to that work because several independently scheduled callers can share one model process.

## No unbounded backlog

Shared serving does not remove the need for application-level freshness policy.

For online robotics, a caller should generally avoid creating an ever-growing queue of stale requests. Each task should define whether it uses:

- latest-only replacement;
- coverage-preserving overlapping windows;
- bounded pending depth;
- deadline-aware cancellation/skipping.

The server schedules submitted work; the ROS-side temporal policy decides which work is still worth submitting or retaining.

## Remote serving is a first-class benefit

The vLLM node should make the endpoint and served model configurable rather than assuming localhost.

```yaml
base_url: http://127.0.0.1:8000/v1
model: cosmos3-edge
```

could later become:

```yaml
base_url: http://dgx-spark:8000/v1
model: larger-video-vlm
```

without changing the ROS-facing result contract.

This allows the same graph to compare:

- local Thor inference;
- a DGX Spark or other LAN server;
- a remote GPU host;
- another OpenAI-compatible serving backend.

Network latency and failure modes must be measured and recorded when serving is remote.

## Relationship to the IPC path

The intended architecture is additive, not a migration mandate.

```text
                    logical request
                          |
             +------------+------------+
             |                         |
             v                         v
      dedicated IPC path         shared API path
             |                         |
       lowest-control-level       multi-caller serving
       runtime integration        model sharing / remote
```

Keeping both paths gives the repository a useful experimental capability: the same saved temporal evidence and task can be run through different runtimes without changing upstream acquisition or downstream evaluation.

## First benchmark: Cosmos3 Edge FlashRT versus vLLM

The most useful immediate experiment is an apples-to-apples comparison using the same Cosmos3 Edge reasoning model rather than comparing different model sizes.

Hold constant:

- exact source frames;
- frame order and temporal span;
- video representation as closely as each backend permits;
- prompt/task;
- output schema;
- maximum output tokens;
- sampling configuration;
- warm/cold state classification.

Compare:

```text
Cosmos3 Edge + FlashRT/IPC
            versus
Cosmos3 Edge + vLLM/API
```

Run first at concurrency=1, then 2, 4, and 8 callers.

The concurrency experiment is essential. The architectural value of a serving runtime may not appear in a single-client benchmark.

## Multiple-caller benchmark shape

One useful synthetic workload is to emulate several independent ROS functions against the same server:

| Caller | Example cadence | Temporal input | Output budget |
| --- | ---: | --- | ---: |
| Slow semantic observer | ~0.25-0.5 Hz | long/sparse window | medium |
| Temporal motion observer | ~0.5-1 Hz | moderate window | compact |
| Weather/visibility observer | slow | sparse frames | compact |
| On-demand diagnostic query | bursty | current evidence | richer |

Measure both isolated and simultaneous execution.

The benchmark should preserve per-request provenance so latency can be attributed to:

- client-side preparation;
- server queue/wait;
- multimodal prefill;
- decode;
- network/HTTP overhead;
- total end-to-end ROS result age.

## Decision criteria

A shared vLLM path is attractive if it provides most of the following:

1. acceptable latency for a single online temporal request;
2. materially better aggregate throughput with multiple callers;
3. predictable p95 latency under expected contention;
4. no duplicated model residency per ROS function;
5. clean structured-output and timing instrumentation;
6. simple model/runtime replacement;
7. ability to serve locally or remotely without changing the ROS contract.

The dedicated IPC path remains preferred for a use case if it delivers materially lower latency, better temporal-input fidelity, or required runtime capabilities that the serving backend cannot match.

## Design invariants

1. ROS/task decomposition remains independent of the model-serving runtime.
2. Backend-specific transport details do not leak into downstream result consumers.
3. Temporal sampling/backpressure remains outside the VLM runtime.
4. Multiple callers may share one model without being forced into one monolithic perception node.
5. No serving backend is assumed superior without concurrency and latency measurements.
6. Runtime/model/input/output provenance is recorded for controlled comparison.
7. The IPC path remains supported while the shared-serving path is evaluated.
