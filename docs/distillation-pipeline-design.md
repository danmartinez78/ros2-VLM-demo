# Temporal VLM task-distillation pipeline design

Date: 2026-08-25 (America/Chicago)  
Status: **target design with implementation scaffold**

## Objective

Build a reproducible task-distillation/post-training pipeline that can specialize a smaller VLM for short-horizon temporal observation while preserving the latency advantage of compact deployment models.

This is task specialization, not generic model compression. The target behavior is narrow:

```text
bounded temporal sequence
        ->
temporal state/change estimate
        ->
compact structured scene observation
```

The initial teacher/student pairing is Cosmos-Reason2-8B -> Cosmos-Reason2-2B, but both roles remain configurable.

## Pipeline stages

```text
raw / controlled temporal sequences
        ->
sample manifest + temporal metadata
        ->
teacher inference
        ->
structured candidate labels
        ->
schema / rule / optional human validation
        ->
accepted canonical dataset
        ->
framework-specific multimodal export
        ->
SFT / QLoRA-style student post-training
        ->
controlled temporal evaluation + latency benchmark
```

Dataset generation, validation, export, training, and evaluation remain independently repeatable stages.

## Central invariant: preserve temporal representation

The same frame bytes are not automatically the same training example when delivered through different temporal representations.

The pipeline must preserve semantic equivalence across teacher inference, canonical provenance, training export, training preprocessing, student evaluation, and target runtime representation.

At minimum, do not silently change:

- frame ordering or count;
- native-video versus independent-image semantics;
- exact timestamps or explicitly declared uniform sampling;
- processor-side sampling/resampling behavior;
- rendered-timestamp controls.

ADR 003 records this decision in [`adr/003-preserve-teacher-student-temporal-representation.md`](adr/003-preserve-teacher-student-temporal-representation.md).

## Canonical temporal sample schema

A representative logical sample is:

```json
{
  "sample_id": "sequence-000123",
  "frames": [
    {"path": ".../frame_000.jpg", "t_seconds": 0.00},
    {"path": ".../frame_001.jpg", "t_seconds": 0.25}
  ],
  "prompt_profile": "temporal_observation_v1",
  "target": {
    "change_detected": true,
    "change": "approaching",
    "state_start": "vehicle distant",
    "state_end": "vehicle closer",
    "evidence_start_s": 0.25,
    "evidence_end_s": 1.75,
    "confidence": 0.92,
    "scene_observation": "vehicle closing distance"
  },
  "provenance": {
    "teacher_model": "Cosmos-Reason2-8B",
    "teacher_engine_identity": "...",
    "teacher_prompt_version": "temporal_teacher_v1",
    "sequence_type": "video",
    "effective_fps": 8.0,
    "timestamp_policy": "uniform_from_fps",
    "rendered_timestamp_control": false,
    "runtime_temporal_encoding": "native_video"
  }
}
```

The canonical format must preserve ordered frames, temporal anchors, compact state/change labels, model/prompt provenance, runtime representation provenance, validation status, and enough information to regenerate model-input semantics.

## Teacher-label generation

Teacher generation should:

1. consume an explicit sequence manifest;
2. preserve chronological ordering and timing;
3. preserve the declared temporal representation;
4. use timestamp overlays or prompt mappings only as explicit controls;
5. request compact structured output rather than free-form reasoning traces;
6. record model/profile/engine provenance;
7. record the actual runtime temporal encoding;
8. keep the raw response for audit/debugging;
9. support restart/resume without regenerating valid cached samples;
10. reject malformed or representation-inconsistent examples.

## Validation and filtering

Before a teacher output enters the training set, validate:

- schema correctness;
- frame/timestamp bounds;
- chronological evidence ordering;
- required fields and confidence range;
- references to nonexistent frames;
- known-static sequence consistency;
- declared temporal representation support;
- timestamp/FPS policy consistency.

Rejected examples remain available with machine-readable rejection reasons.

## Uniform FPS versus exact timestamps

For uniformly sampled sequences, a source/effective FPS can represent spacing. For irregular sequences, average FPS is not equivalent to the actual capture times.

Choose one explicit path:

1. preserve per-frame timestamps through a processor/runtime interface known to consume them correctly;
2. deterministically resample to a uniform sequence and record the transformation; or
3. reject the sample.

Unsupported irregular timing should be rejected rather than silently approximated.

## Training export and processor contract

The exporter converts canonical samples to the selected multimodal training format while preserving frame order, representation, and timing.

For native-video examples:

- already-sampled frames must not be sampled again silently;
- processor timing metadata must represent the source timing actually intended;
- implicit default FPS values must not replace experiment timing;
- full-example and prompt-only masking passes must receive identical video metadata.

These invariants should be covered by CPU-testable preprocessing regressions where possible.

## Student training launcher

A versioned SFT/QLoRA-style launcher should:

- separate model, data, output, and configuration paths;
- support dry-run validation without model downloads;
- save the effective training configuration;
- record base model/checkpoint identity;
- record dataset manifest/hash and preprocessing-library versions;
- save checkpoints/adapters under an experiment ID;
- make resume behavior explicit;
- avoid committed machine-specific absolute paths.

## Evaluation integration

Student evaluation should use the same controlled temporal framework used for model selection. Record at least:

- change-detection accuracy;
- direction/trend accuracy;
- event-time bucket accuracy;
- static false-positive rate;
- schema adherence;
- hallucinated frame-reference rate;
- confidence/calibration when meaningful;
- latency and stage timing;
- effective temporal representation and timing policy.

Controlled transforms should include chronological, reversed, shuffled, repeated-static/duplicated, and single-terminal-frame diagnostics.

## Reproducibility requirements

Every generated dataset, training run, and evaluation artifact should be attributable to:

- repository commit;
- teacher/student model and engine identities;
- prompt version;
- source dataset/manifest hash;
- source frame identities;
- sequence type and runtime temporal encoding;
- timing policy/effective FPS or exact timestamps;
- validation version;
- processor/runtime library versions;
- training configuration;
- final checkpoint/adapter identity.

## Relationship to the temporal runtime

The temporal runtime architecture is documented in [`architecture/temporal-vlm-architecture.md`](architecture/temporal-vlm-architecture.md). Training/evaluation should consume the same representation vocabulary and provenance rather than inventing a parallel contract.

The rolling-window scheduler is upstream of model representation. When live or replayed windows become training examples, their sampling, continuity, and window policies must also be recorded.

## First implementation boundary

The first implementation should build reproducible machinery, not claim a successful specialized model.

In scope:

- canonical sample/dataset schema;
- manifest tooling;
- teacher-generation interface;
- validation/filtering;
- deterministic splitting/export;
- training launcher scaffold;
- evaluation hooks;
- dry-run support;
- CPU-only synthetic tests;
- explicit representation/timing guards.

Out of scope initially:

- large-model execution in CI;
- long hardware training runs;
- quality-improvement claims;
- reinforcement-learning post-training;
- production-specific policy logic.

## Acceptance criteria

The infrastructure is ready for hardware experiments when:

1. ordered-frame datasets convert deterministically into canonical manifests;
2. mocked teacher outputs parse, validate, reject, and resume correctly;
3. invalid frame references are rejected explicitly;
4. accepted samples export into a versioned student dataset;
5. dataset splits and provenance are deterministic;
6. training config validates in dry-run mode;
7. evaluation tooling computes temporal/schema metrics from mock outputs;
8. native-video timing cannot silently fall back to a processor default;
9. prompt-only and full-example preprocessing use identical temporal metadata;
10. unsupported irregular timing is rejected rather than approximated;
11. hardware-independent tests pass;
12. existing model-management and benchmark behavior does not regress.
