# Architecture design map

This directory contains design-level documentation for the temporal VLM and ODD-observation work in this repository.

The existing [`../architecture.md`](../architecture.md) remains the source of truth for the **currently deployed ROS/IPC/Edge-LLM process architecture**. The documents here describe the larger system intent, boundaries between subsystems, and decisions that should remain stable as individual implementations evolve.

## Documents

| Document | Purpose |
| --- | --- |
| [Temporal VLM architecture](temporal-vlm-architecture.md) | End-to-end temporal context path, representation contract, scheduling/runtime boundaries, timing semantics, and evaluation implications |
| [ODD observation system](odd-observation-system.md) | Per-axis estimator architecture, coupled constraints, deterministic exit monitoring, and placement of VLM/CNN/detector/tracker methods |
| [`../distillation-pipeline-design.md`](../distillation-pipeline-design.md) | CR2-8B teacher -> CR2-2B student task-distillation design, with explicit temporal-representation fidelity requirements |

## Architecture decision records

The ADRs in [`../adr/`](../adr/) record decisions that should not be rediscovered from issue history:

1. [`001-native-video-over-ordered-images.md`](../adr/001-native-video-over-ordered-images.md) — use native video semantics for temporal experiments when the runtime supports them.
2. [`002-bounded-window-separate-from-vlm-runtime.md`](../adr/002-bounded-window-separate-from-vlm-runtime.md) — keep live-window scheduling/backpressure separate from VLM temporal encoding.
3. [`003-preserve-teacher-student-temporal-representation.md`](../adr/003-preserve-teacher-student-temporal-representation.md) — preserve temporal representation across teacher generation, training, and evaluation.

## Status language

These documents deliberately distinguish three kinds of statements:

- **Validated/current** — demonstrated on the supported Thor stack or already present on `main`.
- **Implementation in progress** — represented by an open issue or PR and not yet assumed to be on `main`.
- **Target design** — intended architecture that constrains future implementation but may not yet exist.

Do not silently promote a target-design statement into a claim about the current runtime.

## Related tracking

- #8 — bounded rolling temporal-window scheduling and backpressure
- #70 / PR #71 — temporal task-distillation pipeline
- #72 / PR #74 — explicit temporal sequence contract and native video runtime representation
- #75 — architecture documentation tracking issue
