# Thor multi-frame VLM latency characterization

Date: 2026-08-23 (America/Chicago; reports generated 2026-08-24 UTC)

## Research question

Can useful short-horizon temporal context be supplied to a VLM as multiple ordered frames in **one inference** at only modest marginal latency, compared with invoking the VLM independently for each frame?

This experiment characterizes the latency side of that question on Jetson AGX Thor. It does not establish temporal-reasoning quality; task-level quality must be evaluated separately on sequences that require motion/change reasoning.

## Tested configuration

| Item | Configuration |
| --- | --- |
| Hardware | NVIDIA Jetson AGX Thor |
| JetPack / L4T | JetPack 7.2 / R39.2.x |
| Model | `nvidia/Cosmos-Reason2-8B` |
| Quantization | NVFP4 |
| Engine profile | `thor-f8` |
| Max input length | 2048 |
| Max KV cache capacity | 4096 |
| Max image tokens | 2048 |
| Max image tokens per image | 512 |
| Prompt policy | one compact temporal JSON result for the full ordered frame set |
| Max output tokens | 32 |
| Warmup | 1 iteration per condition |
| Measured iterations | 5 per condition |
| Frame conditions | F1, F2, F4, F8 |

The final comparison uses one common managed engine profile for all frame-count conditions so the scaling curve is not confounded by engine-capacity changes.

## Steady-state IPC results

All 20 measured inferences succeeded and output length was held constant.

| Condition | Frames | Mean IPC latency | p95 | Increase vs F1 | Relative to F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | 1 | 708.6 ms | 730.0 ms | - | 1.000x |
| F2 | 2 | 753.8 ms | 759.0 ms | +45.2 ms | 1.064x |
| F4 | 4 | 822.0 ms | 833.0 ms | +113.4 ms | 1.160x |
| F8 | 8 | 957.6 ms | 970.0 ms | +249.0 ms | 1.351x |

Visual context increased **8x** from F1 to F8 while steady-state request latency increased only **35.1%**.

For latency context only, repeating the measured F1 request sequentially would cost approximately:

| Context | Repeated F1 | One multi-frame request | Reduction |
| --- | ---: | ---: | ---: |
| 2 frames | 1417.2 ms | 753.8 ms | 46.8% |
| 4 frames | 2834.4 ms | 822.0 ms | 71.0% |
| 8 frames | 5668.8 ms | 957.6 ms | 83.1% |

Repeated independent inference and one joint multi-frame inference are not semantically identical algorithms; this table compares latency only.

## Native profiled results

NVIDIA `llm_inference --dumpProfile` exposed visual-token count and runtime stage timing:

| Condition | Frames | Visual tokens | Vision encoder | Prefill | Generated tokens | Generation tok/s | Generation GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 1 | 192 | 22.0 ms | 33.3 ms | 31 | 46.8 | 662.4 ms |
| F2 | 2 | 384 | 33.9 ms | 34.1 ms | 31 | 46.7 | 663.2 ms |
| F4 | 4 | 768 | 58.3 ms | 49.3 ms | 31 | 46.2 | 670.6 ms |
| F8 | 8 | 1536 | 102.8 ms | 87.3 ms | 31 | 45.3 | 683.8 ms |

From F1 to F8, the exposed incremental cost was approximately:

```text
vision encoder:    +80.8 ms
prefill:           +54.0 ms
generation GPU:    +21.4 ms
--------------------------------
profiled-stage sum +156.2 ms
```

Generation remains the dominant exposed cost even at F8. Additional frames primarily increase vision encoding and prefill.

## F8 engine-capacity validation

The original control visual engine could not process F8 because its visual-token budget was too small. A managed `thor-f8` engine with `maxImageTokens=2048` was built and successfully handled F8 through both runtime paths.

A direct F8 profile reported roughly:

```text
total_images:            8
total_image_tokens:      1536
vision_encoder:          ~103-114 ms
prefill:                 ~87-92 ms
generation GPU:          ~684-705 ms
generated tokens:        31
```

## Reproducibility lesson

One initial F8 attempt accidentally used stale engine paths. Hard-coded absolute paths to the intended managed engine succeeded, proving the failure was environment/provenance related rather than a model-capacity failure.

Hardware benchmarks should record and print the active model/profile and absolute engine paths. An output-directory name is not evidence of which engine actually ran.

## Interpretation

For short temporal windows on Cosmos-Reason2-8B/Thor, supplying several ordered frames to one inference is substantially more latency-efficient than repeating independent inference for every frame.

F4 is an attractive low-latency point in this test (~822 ms steady state), while F8 provides twice the visual history at ~958 ms and remains below one second mean steady-state latency.

The data also shows that reducing model-generation cost is likely a higher-leverage optimization than aggressively reducing frame count: at F8, generation still dominates the exposed latency profile.

## What this result does not prove

This benchmark isolates latency scaling. It does not establish that F8 is always more accurate or useful than F4/F2.

Temporal-quality experiments should include sequences where the answer actually requires history, for example:

- object motion/direction;
- appearance/disappearance;
- changing weather or visibility;
- approach toward an operating limit or scene boundary;
- transient hazards;
- scene-state changes that cannot be answered reliably from one frame.

## Next experiment

The dominant ~660-684 ms generation term motivated repeating the same F1/F2/F4/F8 matrix with `Cosmos-Reason2-2B` under the same engine/profile, prompt, output-token, and measurement policies.

See [`cosmos-reason2-2b-multiframe-results.md`](cosmos-reason2-2b-multiframe-results.md) and the consolidated [`architecture/temporal-results-matrix.md`](architecture/temporal-results-matrix.md).
