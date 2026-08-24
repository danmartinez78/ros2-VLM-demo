# Architecture decision records

ADRs capture decisions that should remain discoverable even after the implementation details and issue discussions evolve.

| ADR | Decision | Status |
| --- | --- | --- |
| [001](001-native-video-over-ordered-images.md) | Prefer native video semantics for temporal experiments when supported | Accepted |
| [002](002-bounded-window-separate-from-vlm-runtime.md) | Keep bounded rolling-window scheduling separate from VLM runtime representation | Accepted |
| [003](003-preserve-teacher-student-temporal-representation.md) | Preserve temporal representation across teacher generation, student training, evaluation, and target runtime | Accepted |

## ADR format

Each record should include:

- context;
- decision;
- consequences/tradeoffs;
- alternatives considered;
- verification or revisit conditions where applicable.

When a decision changes, prefer adding a superseding ADR rather than silently rewriting the historical decision.
