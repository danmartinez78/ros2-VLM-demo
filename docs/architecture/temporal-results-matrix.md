# Temporal VLM evidence and results matrix

Status: **measured evidence + explicitly open questions**  
Date: 2026-08-25 (America/Chicago)

## Purpose

This document summarizes what the repository has actually demonstrated about multi-frame and native-video VLM inference. It is deliberately model- and application-agnostic.

Detailed results remain in:

- [`../multiframe-temporal-latency-results.md`](../multiframe-temporal-latency-results.md)
- [`../cosmos-reason2-2b-multiframe-results.md`](../cosmos-reason2-2b-multiframe-results.md)
- [`../temporal-chronology-results.md`](../temporal-chronology-results.md)
- [`temporal-vlm-architecture.md`](temporal-vlm-architecture.md)

## Evidence levels

| Evidence level | Meaning |
| --- | --- |
| **Controlled benchmark** | Repeated runs with controlled input/output conditions; suitable for quantitative comparison. |
| **Controlled profile** | Runtime stage/token profiling under a controlled workload. |
| **Controlled semantic test** | Same captured evidence replayed with one deliberately changed semantic variable. |
| **Controlled semantic control** | Negative or perturbation control intended to expose false temporal interpretation. |
| **Hardware validation** | Demonstrates that a runtime/transport contract works on the target hardware. |
| **Smoke test** | Establishes feasibility; not sufficient for a performance or quality claim. |
| **Implementation/API finding** | Verified runtime or processor behavior that constrains experiment representation. |
| **Design invariant** | Rule adopted to prevent known semantic/reproducibility failures. |
| **Open experiment** | Important question for which sufficient evidence has not yet been collected. |

## Results matrix

| Question | Configuration / observation | Result | Evidence level | Current interpretation |
| --- | --- | --- | --- | --- |
| Can CR2-8B ingest a short visual history without latency growing linearly with frame count? | Ordered independent images, F1 to F8, Thor `thor-f8` | Mean IPC latency **708.6 -> 957.6 ms**; 8x frames increased latency **35.1%** | Controlled benchmark | Bounded multi-frame context is much cheaper than repeated independent F1 calls |
| Where does F1-to-F8 cost appear on CR2-8B? | NVIDIA direct profile | Vision **22.0 -> 102.8 ms**; prefill **33.3 -> 87.3 ms**; generation GPU **662.4 -> 683.8 ms** | Controlled profile | Additional frames mainly increase vision/prefill; generation remains dominant |
| Does model size attack the dominant cost? | CR2-8B vs CR2-2B, ordered-image F8 | Mean IPC **957.6 -> 415.4 ms**, **56.6% reduction** | Controlled benchmark | Decoder/model size is a high-leverage latency control |
| Can CR2-2B handle F8? | CR2-2B `thor-f8` | Mean IPC **415.4 ms**, p95 **422 ms** | Controlled benchmark | Eight-frame capacity is established |
| Does ordered multi-image ingestion prove temporal reasoning quality? | Static F8 sanity sequence | No meaningful change reported, but references to nonexistent Image 9/10 appeared | Smoke test | Capacity does not establish temporal grounding |
| Does the explicit sequence contract cause a real runtime representation change? | Same F4 frames through image and video requests | Ordered-image and native-video runtime encodings differ; no fallback | Hardware validation | Temporal representation is a real model-input semantic, not a label |
| Does native video work on Thor? | CR2-2B F4/F8 native video | F4 **559 ms**, F8 **601 ms** inference in smoke runs | Hardware validation | Native video is operational for bounded temporal context |
| Are timing and representation part of model semantics? | Qwen3-VL/Cosmos video path | Native video consumes FPS/timestamp metadata; ordered images do not | Implementation/API finding | Sequence type, runtime encoding, and timing policy are first-class provenance |
| Can irregular timestamps be replaced by average FPS without changing the task? | Runtime/distillation design | **No** | Design invariant | Preserve exact timestamps, explicitly resample and record it, or reject the sample |
| Is Cosmos3 native-video inference sensitive to chronology? | 8-frame, 1.868 s motion-rich window; exact same frames/timestamps forward vs reverse | Forward: person **right -> left**; reverse: **left -> right** | Controlled semantic test | **Chronology sensitivity demonstrated for this sequence** |
| Does repeated static video force a false change? | Final frame repeated 8x with the same native-video timestamp schedule | Structured `CHANGES: none`; summary still described walking | Controlled semantic control | Structured change field passed the static control; free-form summary consistency needs improvement |
| Does the model reject temporally incoherent ordering? | Deterministically shuffled same 8 frames | Model still produced a plausible motion narrative | Controlled semantic control | Temporal coherence/rejection is **not established** |
| Does a single image avoid action inference? | Final frame only | Model inferred walking/forward from appearance | Diagnostic | Treat as action-state inference, not temporal displacement evidence |
| Do we know whether native video improves temporal quality over ordered images across a dataset? | Not yet measured across a shared corpus | **Unknown** | Open experiment | Requires a model-agnostic semantic benchmark |
| Do we know which supported video model has the best quality/latency tradeoff? | Multiple video-capable models are available in the pipeline | **Unknown** | Open experiment | Freeze working baselines and compare on identical saved windows |

## Current conclusions

1. **Short bounded visual histories are computationally tractable on Thor.**
2. **Generation/decoder cost dominates the measured CR2 latency profile.**
3. **Native-video timing and representation must be preserved as experiment provenance.**
4. **Cosmos3 native video has demonstrated chronology sensitivity on a controlled motion sequence.**
5. **General temporal reliability is still open.** Shuffled-order rejection and output-field consistency need dataset-level measurement.
6. **The next step is comparative evaluation across video-capable models**, not additional model-specific prompt tuning on a single clip.

## Next evidence to collect

1. Build a shared set of motion-rich and static saved windows.
2. Run chronological, reversed, shuffled, repeated-static, and dropped-frame controls for each video-capable model.
3. Score direction accuracy, reverse consistency, static false-change rate, shuffle rejection, schema compliance, latency, and memory.
4. Compare native-video and ordered-image representations using identical source frames.
5. Only after comparative evaluation, optimize prompts or specialize/distill the strongest deployment candidates.
