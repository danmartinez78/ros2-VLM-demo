# Cosmos-Reason2-2B multi-frame VLM results on Jetson AGX Thor

Date: 2026-08-23 (America/Chicago; benchmark reports generated 2026-08-24 UTC)

## Purpose

This experiment repeats the F1/F2/F4/F8 multi-frame latency characterization on
`nvidia/Cosmos-Reason2-2B` after the corresponding Cosmos-Reason2-8B experiment.
The goal is to determine whether reducing model size attacks the dominant language-
generation cost while retaining the ability to ingest and reason over a short ordered
frame sequence in one inference.

This is primarily a latency characterization. A small no-change temporal sanity test
is included, but it is not a substitute for a controlled temporal-reasoning quality
benchmark.

## Configuration

| Item | Configuration |
| --- | --- |
| Hardware | NVIDIA Jetson AGX Thor |
| JetPack / L4T | JetPack 7.2 / R39.2.x |
| Model | `nvidia/Cosmos-Reason2-2B` |
| Quantization | NVFP4 |
| Engine profile | `thor-f8` |
| Max input length | 2048 |
| Max KV cache capacity | 4096 |
| Max image tokens | 2048 |
| Max image tokens per image | 512 |
| Decode strategy | standard |
| Prompt policy | compact temporal JSON result for the full ordered frame set |
| Benchmark output limit | 32 tokens |
| Warmup | 1 iteration per condition |
| Measured iterations | 5 per condition |
| Frame conditions | F1, F2, F4, F8 |

The prepared 2B ONNX artifacts were built into a managed engine at:

```text
/home/daniel/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/engines/thor-f8
```

`modelctl status cosmos-reason2-2b thor-f8` reported:

```text
checkpoint_ready: True
quantized_ready: True
onnx_ready: True
engine_ready: True
```

The benchmark report recorded the following managed-engine provenance:

```text
Identity: Cosmos-Reason2-2B/thor-f8@33b69b680207
LLM engine dir: /home/daniel/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/engines/thor-f8/llm
Multimodal dir: /home/daniel/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/engines/thor-f8
Engine manifest: /home/daniel/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/engines/thor-f8/engine-manifest.json
Manifest SHA256: 33b69b680207a9867c4522ccc081509a9868fd42d7a06860e51bd5d7f559068f
Manifest status: matched
```

For this run the active profile was explicitly switched to 2B and the persistent
`edge_vlm_server` was stopped and restarted after activation, so the serving process
was operationally known to have loaded the 2B `thor-f8` engine. PR #68 still has an
open correctness requirement to make IPC benchmark provenance verify the serving
process identity automatically rather than trusting caller-shell state.

## F8 capacity smoke test

Before the full sweep, one eight-frame IPC request succeeded:

```text
success: true
frame_count: 8
inference_seconds: 0.440211
client_latency_ms: 467
```

This establishes that the 2B `thor-f8` visual engine can ingest the same eight-frame
workload used for the 8B experiment.

## Steady-state IPC latency

Report:

```text
/tmp/vlm_multiframe_f1248_cr2_2b_thor_f8/vlm_multiframe_report.txt
```

All 20 measured inferences succeeded.

| Condition | Frames | Mean IPC latency | p95 IPC latency | Increase vs F1 | Relative to F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | 1 | 266.2 ms | 280.0 ms | - | 1.000x |
| F2 | 2 | 249.8 ms | 255.0 ms | -16.4 ms | 0.938x |
| F4 | 4 | 322.2 ms | 331.0 ms | +56.0 ms | 1.210x |
| F8 | 8 | 415.4 ms | 422.0 ms | +149.2 ms | 1.560x |

The small F2-vs-F1 inversion should be treated as run-level variance/output-policy
effects rather than evidence that two frames are intrinsically cheaper than one.
The important trend is that F4/F8 add visual context at modest total latency.

Measured per-run IPC latencies were:

```text
F1: 276, 280, 259, 262, 254 ms
F2: 250, 254, 255, 240, 250 ms
F4: 329, 331, 305, 319, 327 ms
F8: 419, 422, 404, 420, 412 ms
```

## Native profiled stage timing

Report:

```text
/tmp/vlm_multiframe_f1248_direct_cr2_2b_thor_f8/vlm_multiframe_report.txt
```

NVIDIA `llm_inference --dumpProfile` reported:

| Condition | Frames | Visual tokens | Vision encoder | Prefill | Generated tokens | Generation tok/s | Generation GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 1 | 192 | 16.1 ms | 14.6 ms | 32 | 147.0 | 217.8 ms |
| F2 | 2 | 384 | 24.7 ms | 15.9 ms | 32 | 146.7 | 218.1 ms |
| F4 | 4 | 768 | 46.1 ms | 15.8 ms | 32 | 144.0 | 222.2 ms |
| F8 | 8 | 1536 | 72.8 ms | 24.9 ms | 32 | 138.1 | 231.8 ms |

All direct-profile runs reached the configured 32-token maximum, so generation
length is controlled consistently within this direct comparison.

At F8, the sum of exposed profiled stages is approximately:

```text
vision encoder:   72.8 ms
prefill:          24.9 ms
generation GPU:  231.8 ms
-------------------------
exposed sum:     329.5 ms
```

Generation remains the largest exposed stage, but it is dramatically smaller than
with Cosmos-Reason2-8B.

## Cosmos-Reason2-8B vs 2B

Both comparisons use the same `thor-f8` profile limits, NVFP4 quantization, F1/F2/F4/F8
fixture policy, 32-token benchmark limit, one warmup, and five measured iterations.

### Steady-state IPC

| Condition | 8B mean IPC | 2B mean IPC | 2B latency reduction |
| --- | ---: | ---: | ---: |
| F1 | 708.6 ms | 266.2 ms | 62.4% |
| F2 | 753.8 ms | 249.8 ms | 66.9% |
| F4 | 822.0 ms | 322.2 ms | 60.8% |
| F8 | 957.6 ms | 415.4 ms | 56.6% |

At F8, Cosmos-Reason2-2B cuts steady-state IPC latency from approximately 958 ms to
415 ms while preserving the same eight-frame input capacity.

### F8 profiled stages

| Stage | 8B F8 | 2B F8 | Reduction |
| --- | ---: | ---: | ---: |
| Vision encoder | 102.8 ms | 72.8 ms | 29.2% |
| Prefill | 87.3 ms | 24.9 ms | 71.5% |
| Generation GPU | 683.8 ms | 231.8 ms | 66.1% |
| Sum of exposed stages | 873.9 ms | 329.5 ms | 62.3% |

Generation throughput increases from about 45.3 tokens/s on 8B F8 to 138.1 tokens/s
on 2B F8. This strongly supports the earlier hypothesis that model/decoder size is
a higher-leverage latency optimization than aggressively reducing short temporal
frame count.

## Quick output-quality sanity check

The latency fixture contains effectively identical red-panda frames, so a useful
minimal temporal sanity test is whether the model correctly concludes that no
meaningful change occurs across the ordered F8 sequence.

A separate F8 request used a stricter prompt and a 64-token output limit:

```text
Return JSON exactly as:
{"change_detected":true_or_false,"description":"brief description","confidence":0_to_1}
Report only changes that actually occur across the sequence.
```

The 2B model returned:

```json
{"change_detected":false,"description":"Image 3 is identical to Image 6, Image 7 is identical to Image 9, Image 8 is identical to Image 5, Image 10 is identical to Image 4.","confidence":1}
```

Latency for that longer-output quality request was:

```text
inference_seconds: 0.646606
client_latency_ms: 675
frame_count: 8
```

Interpretation:

- **Pass:** the model correctly detected no meaningful temporal change.
- **Pass:** it obeyed the requested compact JSON structure closely enough to be machine-usable.
- **Pass:** it did not invent motion or a scene transition.
- **Known quality issue:** it referenced `Image 9` and `Image 10` despite receiving only eight frames; precise ordinal/frame references are therefore not trustworthy in this sanity test.
- **Earlier warning:** one unconstrained F2 sample hallucinated an `Ocean Waves` object, so basic scene-output quality is not uniformly clean.

The quality sanity test therefore supports initial viability but does **not** establish
2B quality equivalence to 8B.

## Conclusion

The initial Cosmos-Reason2-2B experiment is successful on the latency objective:

> Eight ordered frames can be processed in one Cosmos-Reason2-2B inference on Thor
> at roughly 415 ms mean steady-state IPC latency under the tested 32-token policy,
> versus roughly 958 ms for Cosmos-Reason2-8B using the same managed profile limits.

The native profile confirms that most of the gain comes from attacking the dominant
language-model cost: F8 generation GPU time falls from about 684 ms to 232 ms.

Cosmos-Reason2-2B is therefore a strong candidate for the temporal VLM pipeline, with
a major caveat: quality now needs to be evaluated on controlled sequences containing
real state changes.

## Next experiment: controlled temporal quality

The red-panda fixture is useful for latency and a no-change negative control, but the
next experiment should require temporal evidence.

Recommended sequence classes:

- object appears / disappears;
- person or vehicle approaches / recedes;
- road becomes blocked / unblocked;
- construction-zone entry / exit;
- rain, fog, or visibility worsens / improves;
- ODD-relevant scene state changes that cannot be answered from the terminal frame alone.

Recommended controls:

- chronological sequence;
- reversed sequence;
- shuffled sequence;
- duplicate-frame negative control;
- single terminal-frame baseline.

Compare Cosmos-Reason2-2B and Cosmos-Reason2-8B on:

- change-detection accuracy;
- trend/direction correctness;
- event timing bucket;
- structured-output validity;
- hallucination rate;
- confidence/calibration;
- steady-state latency.

Only after this controlled quality experiment should 2B be treated as a functional
replacement for 8B rather than a latency-optimized candidate.
