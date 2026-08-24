# ADR 003: Preserve teacher/student temporal representation during distillation

- Status: Accepted
- Date: 2026-08-24
- Related: #70 / PR #71, #72 / PR #74, #75

## Context

Task distillation uses a stronger temporal teacher (initially Cosmos-Reason2-8B) to generate labels for a lower-latency student (initially Cosmos-Reason2-2B).

The same source frames can be presented to a VLM through materially different temporal encodings:

- independent ordered images;
- native video from pre-sampled frames;
- encoded video;
- uniform FPS metadata;
- explicit irregular timestamps;
- rendered timestamp text.

If teacher generation, student training, and evaluation silently use different representations, measured student quality no longer reflects the intended teacher behavior. A particularly dangerous failure mode is preserving frame bytes while changing source timing metadata, because the dataset can appear correct while model timestamp tokens are wrong.

## Decision

Treat temporal representation as part of the sample's semantic provenance and preserve it across the distillation lifecycle.

The required invariant is:

```text
teacher temporal semantics
        ~=
training-processor temporal semantics
        ~=
student evaluation temporal semantics
        ~=
target deployment/runtime semantics
```

This does not require identical software implementations, but ordering, timing, and native-video versus independent-image semantics must remain equivalent.

The canonical sample/dataset provenance must record enough information to distinguish two examples that share identical frame bytes but differ in runtime representation or timing policy.

For pre-sampled frame-list training inputs:

- do not allow the processor to re-sample the frame list;
- preserve source/effective FPS as source video metadata, not merely as a frame-sampling request;
- use the same temporal metadata for the full SFT example and the prompt-only pass used for assistant-label masking;
- do not permit an implicit processor default (for example 24 FPS) to replace the recorded source timing.

For irregular timestamps:

- preserve exact timestamps through a processor/runtime path known to consume them correctly; or
- deterministically resample and record the transformation; or
- reject the sample.

Do not silently approximate irregular timing with an average FPS.

## Consequences

### Positive

- Teacher/student quality comparisons remain scientifically interpretable.
- Dataset artifacts can be audited and regenerated.
- Timing bugs become explicit validation failures instead of hidden model-input drift.
- Runtime and training experiments can be grouped by real representation, not only frame filenames.

### Negative / cost

- Dataset provenance and exporter logic become more detailed.
- Processor-version behavior must be verified rather than assumed.
- Some otherwise usable examples must be rejected until an exact timing path is supported.
- Training tests need to cover processor-call metadata, not only schema shape.

## Required provenance

At minimum, temporal samples should retain:

```text
sequence_type
frame timestamps or timestamp policy
effective/source FPS
rendered timestamp control
runtime_temporal_encoding
teacher model/engine identity
prompt version
source manifest/hash
```

Training artifacts additionally need processor/model/checkpoint/config identity.

## Alternatives considered

### Preserve only frame ordering

Rejected because ordering does not preserve native video timing semantics.

### Use processor defaults during training

Rejected because defaults can differ from teacher/runtime timing and silently alter timestamp tokens.

### Approximate all sequences with average FPS

Rejected for irregular sequences because event timing can shift while the data still appears superficially valid.

### Render timestamps into image pixels for all training

Retained as an experimental control, not the default fidelity mechanism. It changes pixels and is not equivalent to native timing metadata.

## Verification expectation

A regression test for an 8 FPS exported sample should prove that the training processor receives metadata that results in 8 FPS temporal semantics and cannot silently become the processor's default timing. The same metadata path must be used for both full-example and prompt-only tokenization used to build SFT labels.
