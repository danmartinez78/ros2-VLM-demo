# Architecture design map

This directory contains design-level documentation for the generic ROS 2 VLM pipeline, including single-frame reasoning, bounded temporal windows, native-video inference, model/runtime isolation, and evaluation.

The existing [`../architecture.md`](../architecture.md) remains the source of truth for the deployed ROS/IPC/runtime process architecture. The documents here describe temporal representation, scheduling boundaries, reproducibility rules, and measured evidence that should remain useful across models and applications.

## Documents

| Document | Purpose |
| --- | --- |
| [Temporal VLM architecture](temporal-vlm-architecture.md) | End-to-end temporal context path, representation contract, scheduling/runtime boundaries, timing semantics, and evaluation implications |
| [Temporal VLM evidence/results matrix](temporal-results-matrix.md) | Compact synthesis of measured latency/runtime evidence, chronology tests, architecture interpretation, and open experiment gaps |
| [Controlled chronology results](../temporal-chronology-results.md) | Exact forward/reverse/shuffled/static control results from the native-video Cosmos3 experiment on Thor |
| [`../distillation-pipeline-design.md`](../distillation-pipeline-design.md) | Teacher-to-student temporal task-distillation design with explicit representation-fidelity requirements |

## Architecture decision records

1. [`001-native-video-over-ordered-images.md`](../adr/001-native-video-over-ordered-images.md) — use native video semantics for temporal experiments when the runtime supports them.
2. [`002-bounded-window-separate-from-vlm-runtime.md`](../adr/002-bounded-window-separate-from-vlm-runtime.md) — keep live-window scheduling/backpressure separate from VLM temporal encoding.
3. [`003-preserve-teacher-student-temporal-representation.md`](../adr/003-preserve-teacher-student-temporal-representation.md) — preserve temporal representation across teacher generation, training, and evaluation.

## Status language

- **Validated/current** — demonstrated on the supported Thor stack or already implemented.
- **Implementation in progress** — represented by an open issue or branch and not yet assumed to be generally available.
- **Target design** — intended architecture that constrains future implementation but may not yet exist.

The results matrix separates controlled benchmarks/profiles, hardware validation, semantic controls, implementation/API findings, design invariants, and open experiments so evidence strength remains explicit.
