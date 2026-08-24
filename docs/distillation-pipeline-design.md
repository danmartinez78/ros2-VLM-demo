# Temporal VLM task-distillation pipeline design

Date: 2026-08-23 (America/Chicago)

## Objective

Build a reproducible task-distillation/post-training pipeline that can specialize
`Cosmos-Reason2-2B` for short-horizon temporal observation while preserving the
latency advantage demonstrated by the existing F1/F2/F4/F8 benchmark.

This is **task specialization**, not generic model compression. The student is not
expected to reproduce arbitrary teacher reasoning. The target behavior is narrow:

```text
ordered frame sequence
        ->
temporal state/change estimate
        ->
compact structured ODD-relevant observation
```

The initial teacher/student pairing is:

```text
teacher: Cosmos-Reason2-8B
student: Cosmos-Reason2-2B
```

Both should remain configurable so stronger teachers or alternate students can be
added later without redesigning the dataset format.

## Pipeline stages

```text
raw / controlled temporal sequences
        |
        v
sample manifest + temporal metadata
        |
        v
teacher inference
        |
        v
structured candidate labels
        |
        v
schema / rule / optional human validation
        |
        v
accepted distillation dataset
        |
        v
SFT / QLoRA-style student post-training
        |
        v
controlled temporal evaluation + latency benchmark
```

The first implementation should make every stage explicit and independently
repeatable. Do not couple dataset generation directly to training.

## Canonical temporal sample schema

A training/evaluation sample should describe the sequence independently from any
specific training framework. Suggested logical shape:

```json
{
  "sample_id": "sequence-000123",
  "frames": [
    {"path": ".../frame_000.jpg", "t_seconds": 0.00},
    {"path": ".../frame_001.jpg", "t_seconds": 0.25}
  ],
  "prompt_profile": "temporal_odd_v1",
  "target": {
    "change_detected": true,
    "change": "approaching",
    "state_start": "vehicle distant",
    "state_end": "vehicle closer",
    "evidence_start_s": 0.25,
    "evidence_end_s": 1.75,
    "confidence": 0.92,
    "odd_observation": "vehicle closing distance"
  },
  "provenance": {
    "teacher_model": "Cosmos-Reason2-8B",
    "teacher_engine_identity": "...",
    "teacher_prompt_version": "temporal_teacher_v1"
  }
}
```

Exact field names may evolve, but the representation must preserve:

- ordered frames;
- explicit temporal anchors;
- compact temporal state/change labels;
- teacher/prompt provenance;
- validation status;
- enough information to regenerate a training example.

## Teacher-label generation

Teacher generation should:

1. consume a sequence manifest rather than discover arbitrary files implicitly;
2. preserve chronological ordering exactly;
3. support visible timestamp overlays and/or timestamp-to-frame prompt mappings;
4. request a compact structured answer rather than free-form chain-of-thought;
5. record teacher model/profile/engine provenance with every generated label;
6. keep the raw teacher response for audit/debugging;
7. support restart/resume without regenerating completed samples;
8. expose deterministic configuration where the runtime permits it.

The pipeline should not silently accept malformed JSON or hallucinated frame/time
references.

## Validation and filtering

Before a teacher output enters the training set, validate at least:

- schema correctness;
- referenced frame/timestamp bounds;
- chronological evidence ordering;
- required fields;
- confidence range;
- no references to nonexistent frames;
- static/no-change consistency where ground truth is known;
- optional task-specific ground-truth checks for controlled sequences.

Rejected examples should remain available with machine-readable rejection reasons.

The current CR2-2B sanity test produced references to `Image 9` and `Image 10` for
an eight-frame input; the validator should explicitly catch this class of error.

## Training dataset export

Keep the canonical dataset independent of the training framework, then provide an
export adapter for the selected SFT/QLoRA implementation.

The exporter should:

- convert canonical samples into the framework's expected multimodal chat/example
  format;
- preserve frame ordering and timestamps;
- generate compact target responses;
- create train/validation/test splits deterministically from a seed;
- prevent related controls from leaking across splits where possible;
- emit a dataset manifest with source sample IDs and hashes.

## Student training launcher

The first student target is `Cosmos-Reason2-2B`.

Provide a versioned training configuration and launcher for an SFT/QLoRA-style
experiment. The launcher should:

- separate model/config/data/output paths;
- support dry-run/config validation without GPU/model downloads;
- save the exact effective training configuration;
- record base model/checkpoint identity;
- record dataset manifest/hash;
- save adapter/checkpoint artifacts under an experiment ID;
- make resume behavior explicit;
- avoid embedding machine-specific absolute paths in committed config.

Do not make GRPO or other reinforcement-learning methods part of the first
implementation. The pipeline should leave room for them later.

## Evaluation integration

A trained student should be evaluable against the same temporal-quality framework
used for model selection.

At minimum capture:

- change-detection accuracy;
- direction/trend accuracy;
- event-time bucket accuracy;
- static-sequence false positives;
- schema adherence;
- hallucinated entity/frame references;
- confidence calibration where labels support it;
- F1/F2/F4/F8 latency;
- vision/prefill/generation timing where available.

Controlled sequence transforms should include chronological, reversed, shuffled,
duplicated, and single-terminal-frame controls.

## Reproducibility requirements

Every generated dataset/training/evaluation artifact should be attributable to:

- repository commit;
- teacher model and engine identity;
- teacher prompt version;
- source dataset/manifest hash;
- filtering/validation version;
- student base checkpoint;
- training configuration;
- final adapter/checkpoint identity.

The benchmark provenance work in PR #68 is the model for how runtime identity
should be handled: labels such as a directory name are not sufficient proof of what
actually ran.

## First implementation boundary

The first PR for this pipeline should build the **reproducible machinery**, not
claim a successful distilled model.

In scope:

- canonical sample/dataset schema;
- manifest tooling;
- teacher-generation command interface;
- label parsing/validation/filtering;
- deterministic dataset splitting/export;
- student training config/launcher scaffold;
- evaluation adapter/hooks;
- dry-run support;
- CPU-only unit/integration tests using tiny synthetic fixtures;
- documentation and example commands.

Out of scope for the first PR:

- downloading/running large models in CI;
- a long Thor training run;
- claiming quality improvement;
- GRPO/RL post-training;
- automatic human-review UI;
- production runtime escalation logic.

## Acceptance criteria for the infrastructure PR

The implementation is ready for hardware use when:

1. a synthetic ordered-frame dataset can be converted into the canonical manifest;
2. mocked teacher responses can be parsed, validated, accepted/rejected, and
   resumed deterministically;
3. invalid frame references such as `Image 10` for an eight-frame sample are
   rejected with an explicit reason;
4. accepted samples can be exported into a versioned student-training dataset;
5. train/validation/test splitting is deterministic and provenance is recorded;
6. the student-training launcher validates configuration and produces a dry-run
   plan without downloading or loading a model;
7. evaluation tooling can consume a mock trained-student output and compute the
   temporal/schema metrics;
8. all new hardware-independent tests pass in CI;
9. no existing benchmark/model-management behavior regresses.

The first real hardware experiment can then use a small controlled dataset to prove
end-to-end teacher generation -> filtering -> QLoRA/SFT -> evaluation before scaling
up dataset generation or training duration.
