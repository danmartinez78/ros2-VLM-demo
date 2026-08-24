# Temporal VLM task-distillation pipeline design

Date: 2026-08-24 (America/Chicago)  
Status: **target design with implementation in PR #71**  
Tracking: #70, PR #71, #72 / PR #74, #75

## Objective

Build a reproducible task-distillation/post-training pipeline that can specialize
`Cosmos-Reason2-2B` for short-horizon temporal observation while preserving the
latency advantage demonstrated by the existing F1/F2/F4/F8 benchmark.

This is **task specialization**, not generic model compression. The student is not
expected to reproduce arbitrary teacher reasoning. The target behavior is narrow:

```text
bounded temporal sequence
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

Both remain configurable so stronger teachers or alternate students can be added
later without redesigning the canonical dataset format.

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
accepted canonical distillation dataset
        |
        v
framework-specific multimodal export
        |
        v
SFT / QLoRA-style student post-training
        |
        v
controlled temporal evaluation + latency benchmark
```

Every stage should remain explicit and independently repeatable. Dataset generation
must not be coupled directly to training.

## Central invariant: preserve temporal representation

The same frame bytes are **not automatically the same training example** when they
are delivered through different temporal representations.

The pipeline must preserve semantic equivalence across:

```text
teacher temporal representation
        ~=
canonical sample provenance
        ~=
training-export representation
        ~=
training-processor representation
        ~=
student evaluation representation
        ~=
target runtime representation
```

The equality is semantic rather than byte-for-byte. At minimum, these properties
must not change silently:

- frame ordering;
- frame count;
- native-video versus independent-image semantics;
- source/effective FPS for uniformly sampled sequences;
- explicit frame timestamps when exact irregular timing is required;
- rendered-timestamp controls;
- processor-side frame sampling/resampling behavior.

ADR 003 records this decision in [`adr/003-preserve-teacher-student-temporal-representation.md`](adr/003-preserve-teacher-student-temporal-representation.md).

## Canonical temporal sample schema

A training/evaluation sample should describe the sequence independently from any
specific training framework. A representative logical shape is:

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
    "teacher_prompt_version": "temporal_teacher_v1",
    "sequence_type": "video",
    "effective_fps": 8.0,
    "timestamp_policy": "uniform_from_fps",
    "rendered_timestamp_control": false,
    "runtime_temporal_encoding": "native_qwen3vl_video_imagedata_mrope_timestamps"
  }
}
```

Exact field names may evolve, but the representation must preserve:

- ordered frames;
- explicit temporal anchors;
- compact temporal state/change labels;
- teacher/prompt provenance;
- runtime representation provenance;
- validation status;
- enough information to regenerate the model input semantics.

Two samples containing identical JPEG paths but different `sequence_type`, timing,
or runtime temporal encoding must be treated as different experiment examples.

## Teacher-label generation

Teacher generation should:

1. consume a sequence manifest rather than discover arbitrary files implicitly;
2. preserve chronological ordering exactly;
3. preserve the declared native temporal representation and timing metadata;
4. allow rendered timestamp overlays or timestamp-to-frame prompt mappings only as
   explicit experimental controls;
5. request a compact structured answer rather than free-form chain-of-thought;
6. record teacher model/profile/engine provenance with every generated label;
7. record the effective temporal representation used for the forward pass;
8. keep the raw teacher response for audit/debugging;
9. support restart/resume without regenerating valid completed samples;
10. expose deterministic configuration where the runtime permits it.

The pipeline must not silently accept malformed JSON, hallucinated frame/time
references, or a runtime representation different from the one recorded in sample
provenance.

## Validation and filtering

Before a teacher output enters the training set, validate at least:

- schema correctness;
- referenced frame/timestamp bounds;
- chronological evidence ordering;
- required fields;
- confidence range;
- no references to nonexistent frames;
- static/no-change consistency where ground truth is known;
- optional task-specific ground-truth checks for controlled sequences;
- declared temporal representation is supported by the exporter/training path;
- timestamp/FPS policy is internally consistent.

Rejected examples should remain available with machine-readable rejection reasons.

The initial CR2-2B sanity test produced references to `Image 9` and `Image 10` for
an eight-frame input; the validator should explicitly catch this class of error.

## Uniform FPS versus explicit timestamps

For a uniformly sampled sequence:

```text
t_i = t_0 + i / fps
```

A single source/effective FPS value can faithfully represent the temporal spacing.

For an irregular sequence, an average FPS is not equivalent to the actual capture
times. If exact timing is relevant, the pipeline must choose one of three explicit
paths:

1. preserve per-frame timestamps through a processor/runtime interface known to
   consume them correctly;
2. deterministically resample the source to a uniform sequence, record the
   transformation, and train on that new sequence; or
3. reject the sample from the experiment.

The first implementation should reject unsupported irregular timing rather than
silently approximate it.

## Training dataset export

Keep the canonical dataset independent of the training framework, then provide an
export adapter for the selected SFT/QLoRA implementation.

The exporter should:

- convert canonical samples into the framework's expected multimodal chat/example
  format;
- preserve frame ordering and temporal representation;
- preserve source/effective FPS or explicit timestamps according to the sample's
  timing policy;
- generate compact target responses;
- create train/validation/test splits deterministically from a seed;
- prevent related controls from leaking across splits where possible;
- emit a dataset manifest with source sample IDs, content hashes, and representation
  provenance.

### Native-video export

For samples generated/evaluated through native Qwen3-VL/Cosmos-Reason2 video
semantics, the training export should use the processor-native video representation
rather than N independent image entries.

A pre-sampled frame-list example may logically look like:

```json
{
  "type": "video",
  "path": ["frame0.jpg", "frame1.jpg", "frame2.jpg"],
  "fps": 8.0
}
```

The exact processor API may require timing to be supplied separately as
`video_metadata`; the canonical dataset must retain enough information to construct
that metadata correctly.

## Training processor contract

This boundary is easy to get superficially correct while changing the actual model
semantics, so it is an explicit design contract.

### Pre-sampled frames are already selected

If the exported video content is a list of already-sampled frame paths, the
training processor must not silently select a different set of frames.

Conceptually:

```text
do_sample_frames = false
```

or the equivalent supported API is required.

### Sampling FPS is not necessarily source metadata

A processor argument named `fps` may control how frames are sampled from an encoded
video without populating the source timing metadata used later to create model
timestamp tokens.

Therefore, the training implementation must verify the processor behavior for the
specific Transformers/Qwen3-VL version in use. Source timing must be represented in
the metadata path that the processor actually consumes when constructing temporal
features/tokens.

Do not assume that passing `fps=8` while disabling sampling proves that the model
receives 8 FPS timing semantics.

### Avoid implicit default timing

If the processor would otherwise synthesize a default FPS (for example 24 FPS), an
8 FPS training sample must not silently become a 24 FPS sample.

A CPU-testable regression should prove that the metadata derived from an exported
8 FPS example resolves to 8 FPS temporal semantics through the preprocessing path.

### Full example and prompt-only masking pass must match

The SFT launcher may process the example twice:

1. full user + assistant example;
2. prompt-only prefix to determine which tokens should be masked from the loss.

Both calls must receive identical video timing/sampling metadata. Otherwise the
prompt token sequence can differ and `prompt_len` can mask the wrong assistant
labels.

## Student training launcher

The first student target is `Cosmos-Reason2-2B`.

Provide a versioned training configuration and launcher for an SFT/QLoRA-style
experiment. The launcher should:

- separate model/config/data/output paths;
- support dry-run/config validation without GPU/model downloads;
- save the exact effective training configuration;
- record base model/checkpoint identity;
- record dataset manifest/hash;
- record processor/model library versions relevant to temporal preprocessing;
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
- confidence/calibration where labels support it;
- F1/F2/F4/F8 latency;
- vision/prefill/generation timing where available;
- effective temporal representation and timing policy used during evaluation.

Controlled sequence transforms should include chronological, reversed, shuffled,
duplicated, and single-terminal-frame controls.

Evaluation controls must not accidentally change the representation in an
unrecorded way. For example, reversing the semantic order of frames is an intended
control; silently changing FPS while doing so is not.

## Reproducibility requirements

Every generated dataset/training/evaluation artifact should be attributable to:

- repository commit;
- teacher model and engine identity;
- teacher prompt version;
- source dataset/manifest hash;
- source frame identities;
- sequence type;
- timing/timestamp policy and effective FPS;
- runtime temporal encoding;
- filtering/validation version;
- student base checkpoint;
- processor/Transformers version where preprocessing semantics matter;
- training configuration;
- final adapter/checkpoint identity.

The benchmark provenance work in PR #68 is the model for how runtime identity
should be handled: directory names or human labels are not sufficient proof of what
actually ran.

## Relationship to the temporal runtime

The temporal runtime architecture is documented in
[`architecture/temporal-vlm-architecture.md`](architecture/temporal-vlm-architecture.md).

PR #74 establishes the request/runtime representation contract used as the target
for native temporal inference. The distillation pipeline should consume that
contract as provenance rather than inventing a parallel vocabulary.

The rolling-window scheduler in #8 is upstream of both teacher and student runtime
representation. If a training dataset is generated from live/replayed windows, its
sampling/window policy must also be recorded.

## First implementation boundary

The first implementation PR for this pipeline should build the **reproducible
machinery**, not claim a successful distilled model.

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
- documentation and example commands;
- explicit temporal representation/timing guards.

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
8. an 8 FPS native-video sample cannot silently become the processor's default FPS
   during student preprocessing;
9. full-example and prompt-only tokenization use identical temporal metadata;
10. unsupported irregular timing is rejected rather than approximated;
11. all new hardware-independent tests pass in CI;
12. no existing benchmark/model-management behavior regresses.

The first real hardware experiment can then use a small controlled dataset to prove
end-to-end teacher generation -> filtering -> QLoRA/SFT -> evaluation before scaling
up dataset generation or training duration.
