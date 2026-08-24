# Temporal VLM evidence and results matrix

Status: **measured evidence + explicitly open questions**  
Date: 2026-08-24 (America/Chicago)

## Purpose

This document is a compact synthesis of the temporal-VLM evidence collected so far. It is intentionally separate from the detailed benchmark reports and from the architecture documents themselves.

The goal is to answer two questions quickly:

1. **What have we actually demonstrated?**
2. **What architectural conclusion, if any, does each result support?**

Detailed raw methodology and per-run measurements remain in:

- [`../multiframe-temporal-latency-results.md`](../multiframe-temporal-latency-results.md) — Cosmos-Reason2-8B F1/F2/F4/F8 ordered-multi-image latency and profiling;
- [`../cosmos-reason2-2b-multiframe-results.md`](../cosmos-reason2-2b-multiframe-results.md) — Cosmos-Reason2-2B repetition and 8B-vs-2B comparison;
- [`temporal-vlm-architecture.md`](temporal-vlm-architecture.md) — architecture interpretation and future experiment design.

## Evidence levels

| Evidence level | Meaning |
| --- | --- |
| **Controlled benchmark** | Repeated runs with controlled frame condition and output policy; suitable for quantitative latency comparison. |
| **Controlled profile** | NVIDIA runtime stage/token profiling under a controlled workload. |
| **Hardware validation** | Demonstrates that an implementation/runtime contract works on Jetson AGX Thor. |
| **Smoke / sanity test** | Establishes feasibility or catches gross failures; not sufficient for a performance or quality claim. |
| **Implementation/API finding** | Verified behavior of the runtime or processor that constrains how experiments/training must be represented. |
| **Design invariant** | Architectural rule adopted to prevent known semantic/reproducibility failures. |
| **Open experiment** | Important question for which sufficient evidence has not yet been collected. |

## Results matrix

| Question | Configuration / observation | Result | Evidence level | Current interpretation |
| --- | --- | --- | --- | --- |
| Can CR2-8B ingest a short visual history without latency growing linearly with frame count? | Ordered independent images, F1 to F8, Thor `thor-f8` | Mean IPC latency: **708.6 ms -> 957.6 ms**. Eight times as many frames increased latency by **35.1%**. | Controlled benchmark | A bounded multi-frame context is dramatically cheaper than repeated independent F1 calls; frame count alone is not the dominant latency driver. |
| Where does the F1-to-F8 cost appear on CR2-8B? | NVIDIA direct profile | Vision: **22.0 -> 102.8 ms**; prefill: **33.3 -> 87.3 ms**; generation GPU: **662.4 -> 683.8 ms**. | Controlled profile | Extra frames mainly add vision and prefill work; decode/generation remains the dominant stage. |
| Does reducing model size attack the dominant cost? | CR2-8B vs CR2-2B, ordered-image F8 | Mean IPC latency: **957.6 -> 415.4 ms**, a **56.6% reduction**. | Controlled benchmark | Yes. Model/decoder size is a higher-leverage latency control than aggressively shrinking an already-bounded visual context. |
| How much faster is CR2-2B generation at F8? | NVIDIA direct profile | Generation GPU: **683.8 -> 231.8 ms**; throughput: **45.3 -> 138.1 tokens/s**. | Controlled profile | The 2B deployment target directly attacks the largest exposed latency component. |
| Can CR2-2B handle the same ordered-image F8 workload? | CR2-2B `thor-f8` | F8 succeeds; mean IPC **415.4 ms**, p95 **422 ms**. | Controlled benchmark | Eight-frame capacity is established for the 2B engine profile. |
| Does ordered multi-image ingestion prove temporal reasoning quality? | CR2-2B static F8 sanity sequence | Model correctly reported no meaningful change but hallucinated references to nonexistent `Image 9` and `Image 10`. | Smoke / sanity test | Capacity and coarse no-change detection do not establish reliable temporal semantics or frame-reference grounding. |
| Did PR #74 actually switch the runtime representation rather than relabel ordered images? | Same F4 source frames through `images` and `video` requests | `images` reports `ordered_multi_image_no_native_temporal_metadata`; `video` reports `native_qwen3vl_video_imagedata_mrope_timestamps`; both report `temporal_fallback_used=false`. | Hardware validation | The sequence contract causes a real representation change in the Edge-LLM/Qwen3-VL path. |
| Does native video work at F4 on Thor? | CR2-2B, 4 frames, `video`, 4 FPS | Success; **559 ms** inference, **577 ms** client latency; native temporal encoding; no fallback. | Smoke / hardware validation | Native Qwen3-VL/Cosmos-Reason2 video semantics are operational on Thor. |
| Does native video scale to F8? | CR2-2B, 8 frames, `video`, 8 FPS | Success; **601 ms** inference, **635 ms** client latency; native temporal encoding; no fallback. | Smoke / hardware validation | The current native-video implementation handles the targeted F8 short-history case. |
| Does doubling native-video F4 to F8 cause a large latency jump? | F4 vs F8 native-video smoke runs | Inference increased from **559 ms to 601 ms** (~**7.4%**); client latency from **577 ms to 635 ms** (~**10.1%**). | Smoke test | Encouraging scaling result, but not yet a controlled benchmark because generated outputs differed. |
| Is native video faster than ordered images? | F4 smoke comparison, same source frames and nominal prompt | Native video: **559 ms** inference; ordered images: **574 ms**. | Smoke test only | No performance claim. Output lengths differed, so the ~15 ms difference is not controlled. |
| Are timing and temporal representation part of model semantics? | Qwen3-VL/Cosmos-Reason2 video path and distillation investigation | Native video carries FPS/timestamps into model-specific temporal processing; ordered images do not. | Implementation/API finding | `sequence_type`, runtime temporal encoding, and timing policy are first-class experiment provenance. |
| Is passing `fps=8` sufficient to preserve source timing for pre-sampled training frames? | Hugging Face/Qwen3-VL processor investigation for PR #71 | No. Sampling FPS and the `video_metadata.fps` consumed for temporal timestamp construction are not interchangeable. | Implementation/API finding | The training path must construct processor-native timing metadata explicitly; otherwise an 8 FPS example can silently acquire default timing semantics. |
| Can irregular timestamps be represented by an average FPS without changing the task? | Distillation/runtime design | No. | Design invariant | Preserve exact timestamps, explicitly resample to a uniform sequence and record it, or reject the sample. |
| Do we know whether native-video input improves temporal task quality over ordered images? | Not yet tested with controlled motion/change tasks | **Unknown.** | Open experiment | Requires chronological/reversed/shuffled/duplicate/terminal-only controls on real change tasks. |
| Do we know native-video visual-token scaling for F4/F8/F15? | Not yet profiled after #74 | **Unknown.** | Open experiment | Measure native-video token counts before deciding whether a larger visual-engine token budget/profile is necessary. |
| Do we know whether CR2-2B can recover enough temporal quality through task distillation? | PR #71 currently builds the reproducible training scaffold | **Unknown.** | Open experiment | Hardware training/evaluation is required after the processor representation-fidelity path is correct. |

## Current conclusions

1. **Short visual history is comparatively cheap.** F8 does not cost anything close to eight independent F1 calls.
2. **Generation dominates latency**, especially for Cosmos-Reason2-8B. Additional temporal frames primarily increase vision and prefill cost.
3. **Cosmos-Reason2-2B is therefore a strong deployment candidate** if task-specific quality can be retained or recovered through specialization/distillation.
4. **Native video semantics are operational on Thor** and are now the preferred baseline representation for experiments that claim temporal/video semantics.
5. **Representation and timing are part of the experiment**, not incidental preprocessing settings. Identical frame bytes delivered through different temporal encodings are different experimental conditions.
6. **Temporal reasoning quality is not yet established.** The next major evidence gap is controlled semantic evaluation, not another basic capacity smoke test.

## Next evidence to collect

Priority measurements should close the gaps above rather than repeat already-settled feasibility questions:

1. controlled native-video F4/F8/F15 profile: visual tokens, vision, prefill, generation, end-to-end latency;
2. native-video versus ordered-images semantic A/B on appears/disappears, approaches/recedes, weather change, blocked/unblocked road, and construction-zone transitions;
3. chronological versus reversed versus shuffled versus duplicate versus terminal-only controls;
4. CR2-8B teacher versus CR2-2B baseline versus distilled CR2-2B on the same task set;
5. calibration/static false-positive analysis alongside latency so a faster model is not selected on throughput alone.

Treat each new row added to this matrix as evidence only after its measurement method and provenance are recorded in the corresponding detailed experiment artifact.
