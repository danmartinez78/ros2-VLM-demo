# Temporal VLM next-experiment plan

Date: 2026-08-25 (America/Chicago)

## Purpose

The initial multi-frame and native-video work established that short temporal context is tractable on Jetson AGX Thor and that frame ordering can materially change Cosmos3 motion interpretation.

The next question is no longer simply "how many frames can we afford?" It is:

> Which temporal representation and video-capable model provides the best quality/latency tradeoff on the same controlled evidence?

This roadmap is intentionally application-agnostic. Experiments should use saved frame windows, exact timestamps, repeatable prompts, normalized outputs, and identical scoring rules across models.

## Baselines to preserve

1. Single-frame VLM inference.
2. Ordered independent multi-image inference.
3. Native-video inference with exact timestamp metadata.
4. Optional recurrent text/history context.
5. Optional detector/tracker context upstream of the VLM.

These are separate experimental conditions and should not be conflated.

## Shared temporal controls

Every video-capable model should be evaluated on the same saved windows with:

- chronological order;
- reversed order with the same timestamp schedule;
- deterministic shuffled order;
- repeated-static frames;
- dropped/sparse frames;
- a single terminal-frame diagnostic.

The controlled chronology harness in `flashrt_temporal/temporal_chronology_test.py` already produces the first four controls plus the single-frame diagnostic.

## Core metrics

For each model/representation, record:

- motion/change direction accuracy;
- reverse consistency;
- static false-change rate;
- shuffled-sequence rejection or uncertainty;
- camera-motion accuracy when ground truth is available;
- structured-output/schema compliance;
- contradiction rate between structured fields and free-form text;
- inference latency and client latency;
- memory/engine footprint;
- practical maximum frame/window size.

## Recommended sequence

### 1. Build a shared temporal corpus

Select a small set of motion-rich and static windows from existing rosbags. Save exact frames and timestamps so every model sees identical evidence.

### 2. Compare video-capable models

Run the same controls through each supported native-video model before doing model-specific prompt tuning. This establishes which models are actually worth deeper optimization.

### 3. Compare representations

For models that support both modes, compare native video against ordered independent images using the same frames and output contract.

### 4. Evaluate temporal memory alternatives

Only after the direct-video baseline is measured, compare recurrent text summaries, detector/tracker context, structured scene memory, or other memory mechanisms.

### 5. Specialize only strong candidates

Use task distillation or lightweight fine-tuning only after a model/representation shows a promising baseline quality/latency tradeoff.

## Evidence discipline

A result should be added to the evidence matrix only when the input frames, timing policy, representation, model/engine identity, prompt version, output limit, and measurement method are recorded.

See:

- [`architecture/temporal-results-matrix.md`](architecture/temporal-results-matrix.md)
- [`temporal-chronology-results.md`](temporal-chronology-results.md)
- [`architecture/temporal-vlm-architecture.md`](architecture/temporal-vlm-architecture.md)
