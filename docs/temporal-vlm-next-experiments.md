# Temporal VLM next-experiment plan

Date: 2026-08-23 (America/Chicago)

## Purpose

The initial multi-frame latency work established that short temporal context can be
bundled into one VLM request efficiently on Jetson AGX Thor, and that model size is
a much larger latency lever than frame count for the tested workload.

The immediate question is no longer just "how many frames can we afford?" It is:

> What is the best way to recover temporal-reasoning quality while preserving the
> latency advantage demonstrated by Cosmos-Reason2-2B?

This document captures candidate experiment branches and their recommended order.
It is intentionally a planning/roadmap document rather than a set of implementation
issues. Individual experiments should become GitHub issues only after we select them
and can define concrete inputs, acceptance criteria, artifacts, and comparison rules.

## Current baseline

All results below used the common managed `thor-f8` profile with F1/F2/F4/F8
ordered-frame requests.

| Model | F1 IPC mean | F2 IPC mean | F4 IPC mean | F8 IPC mean |
| --- | ---: | ---: | ---: | ---: |
| Cosmos-Reason2-8B NVFP4 | 708.6 ms | 753.8 ms | 822.0 ms | 957.6 ms |
| Cosmos-Reason2-2B NVFP4 | 266.2 ms | 249.8 ms | 322.2 ms | 415.4 ms |

At F8, the 2B model reduced steady-state IPC latency by approximately **56.6%**.

The profiled generation stage shows why:

| Model | F8 vision | F8 prefill | F8 generation GPU |
| --- | ---: | ---: | ---: |
| Cosmos-Reason2-8B NVFP4 | 102.8 ms | 87.3 ms | 683.8 ms |
| Cosmos-Reason2-2B NVFP4 | 72.8 ms | 24.9 ms | 231.8 ms |

The 2B model therefore gives substantial latency headroom for either higher-quality
inference or selective escalation to a larger model.

## Current quality signal

The 2B model passed a basic static-sequence sanity check:

- eight identical frames were accepted in one request;
- the model correctly returned `change_detected=false`;
- it did not invent a scene transition or motion;
- it followed the requested compact JSON structure closely enough to be usable.

However, quality limitations were also visible:

- one short benchmark response hallucinated an `Ocean Waves` object;
- the temporal sanity response referred to `Image 9` and `Image 10` even though only
  eight images were supplied;
- the static fixture does not test real temporal reasoning such as direction,
  appearance/disappearance, worsening visibility, or event timing.

The latency result is therefore strong, while temporal quality remains an open
engineering question.

## Candidate experiment branches

### A. Improve the existing 2B path before changing model families

This is the cheapest branch and should be done first because it can separate model
capacity limits from prompt/representation/quantization effects.

#### A1. Timestamped frame representation

Add explicit time information to each ordered frame, preferably as visible overlays
and/or a consistent prompt mapping:

```text
frame_1 -> t=0.00 s
frame_2 -> t=0.25 s
...
frame_8 -> t=1.75 s
```

Use a compact temporal output such as:

```json
{
  "change_detected": true,
  "change": "approaching",
  "evidence_start_s": 0.50,
  "evidence_end_s": 1.75,
  "confidence": 0.87
}
```

Goal: determine whether explicit temporal anchors reduce ordinal/indexing errors and
improve localization without retraining.

#### A2. Higher-precision Cosmos-Reason2-2B

Build a higher-precision 2B engine if supported by the current preparation/runtime
workflow and repeat the same quality and latency matrix.

Goal: determine whether some observed hallucination/instability is caused by NVFP4
rather than model capacity.

Decision question:

> Can a modest precision/latency increase materially improve temporal correctness
> while keeping F8 comfortably below the 8B baseline?

### B. Larger Qwen-family VLM with speculative decoding

The second branch tests whether a larger general VLM can recover quality while
speculative decoding offsets the additional generation cost.

The preferred first candidate is a mid-size model rather than jumping immediately
to the largest available option.

Suggested comparison structure:

```text
Cosmos-Reason2-2B
  small / fast / standard decoding

vs

Qwen-family ~4B class
  larger model / vanilla decoding
  larger model / speculative decoding
```

If the mid-size model clearly improves temporal quality but still leaves important
failure modes, then test a larger Qwen variant with the same speculative strategy.

Primary question:

> Can speculative decoding move us upward on the reasoning-quality curve without
> giving back most of the 2B latency win?

Measurements should include:

- F1/F2/F4/F8 steady-state IPC latency;
- vision, prefill, and generation timing;
- generation tokens/s;
- speculative acceptance metrics where available;
- memory footprint;
- temporal task accuracy on controlled sequences.

### C. Task-specialize Cosmos-Reason2-2B

If the 2B model remains attractive after the cheap representation/precision tests,
specialization may be more valuable than moving permanently to a larger model.

The target should be **task distillation/post-training**, not generic model
compression.

The desired student behavior is narrow:

```text
ordered frame sequence
        ->
temporal state/change estimate
        ->
structured ODD-relevant observation
```

A candidate training pipeline is:

```text
controlled temporal sequence
        |
        v
strong teacher model
        |
        +-- state at start/end
        +-- detected change
        +-- direction/trend
        +-- temporal evidence
        +-- confidence
        +-- ODD-relevant interpretation
        |
        v
rule/human validation
        |
        v
2B SFT/QLoRA-style post-training dataset
```

The student should be trained toward compact structured outputs rather than verbose
chain-of-thought-style responses. Short outputs improve both determinism and
latency.

### D. Fast-path + escalation architecture

The experiments above may ultimately support a hybrid runtime rather than a single
winner:

```text
frames
  |
  v
specialized / fast 2B temporal observer
  |
  +-- confident, routine case -----------------> deterministic ODD monitor
  |
  +-- low confidence / contradictory evidence
      / near-boundary / novel condition
                         |
                         v
                 larger VLM + speculative decoding
                         |
                         v
                 deterministic ODD monitor
```

This architecture would aim for 2B-class average latency while retaining a larger
model for ambiguous long-tail cases.

Because the current ODD-exit work is focused on detection/logging rather than direct
safety actuation, this escalation strategy can be evaluated without requiring every
observation to take the large-model path.

## Controlled temporal-quality dataset

Before making a model-selection decision, create a small but deliberate evaluation
set where the answer actually depends on temporal ordering.

Include at least:

- static/no-change negative controls;
- object appears;
- object disappears;
- object approaches;
- object recedes;
- left/right or crossing motion;
- road becomes blocked/unblocked;
- visibility worsens/improves;
- rain/fog intensity changes;
- construction-zone transition;
- transient hazard appears briefly;
- progression toward an ODD boundary.

For each sequence, add controls:

- chronological order;
- reversed order;
- shuffled order;
- duplicated-frame sequence;
- single terminal-frame baseline.

Useful metrics:

- change-detection accuracy;
- direction/trend accuracy;
- event-timing bucket accuracy;
- false-positive rate on static sequences;
- schema adherence;
- hallucinated entities/frame references;
- confidence calibration;
- latency.

## Recommended order

The proposed sequence is:

1. **Timestamp/temporal-anchor experiment on Cosmos-Reason2-2B.**
2. **Higher-precision Cosmos-Reason2-2B**, if compatible with the current toolchain.
3. Build the **controlled temporal-quality evaluation set** so later model comparisons
   measure more than visual sanity.
4. Test a **mid-size Qwen-family VLM** with vanilla and speculative decoding.
5. Test a larger Qwen variant only if the mid-size model materially improves quality
   but still leaves important errors.
6. If 2B remains competitive, begin **task-specific post-training/distillation**.
7. Evaluate the **2B fast-path + larger-model escalation** architecture.

This ordering deliberately postpones training work until we know whether inexpensive
representation/precision changes or a larger speculative model already solve the
quality gap.

## Decision criteria

The target is not the highest standalone model score. The target is the best system
for temporal ODD observation on Thor.

A candidate is compelling if it improves temporal correctness while preserving most
of the practical advantages demonstrated by the 2B baseline:

- sub-second F8 steady-state latency;
- predictable structured output;
- low hallucination rate;
- useful temporal ordering/direction reasoning;
- manageable memory and engine complexity;
- reproducible engine/profile provenance.

The 2B result changes the optimization problem: there is now enough latency headroom
to trade some speed for quality intentionally rather than accepting the 8B latency
as the starting point.

## When to create GitHub issues

Do not create one issue per idea in this document yet.

Create an issue when an experiment is selected and can be stated as an executable
piece of work with:

- exact model/profile;
- fixture/evaluation dataset;
- benchmark commands;
- metrics;
- expected artifacts;
- acceptance/comparison criteria.

For example, after selecting the Qwen experiment, an issue could be scoped as:

> Build and benchmark `<selected-model>` with standard and speculative decoding on
> Thor using the established F1/F2/F4/F8 latency harness plus the controlled temporal
> quality set, and compare it against Cosmos-Reason2-2B/thor-f8.

That is issue-sized work; the broader choice among models, precision, distillation,
and escalation is better kept in this roadmap document.
