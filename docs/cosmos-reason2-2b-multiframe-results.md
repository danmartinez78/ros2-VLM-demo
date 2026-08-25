# Cosmos-Reason2-2B multi-frame VLM results on Jetson AGX Thor

Date: 2026-08-23 (America/Chicago; reports generated 2026-08-24 UTC)

## Purpose

This experiment repeats the F1/F2/F4/F8 multi-frame latency characterization on `nvidia/Cosmos-Reason2-2B` after the corresponding Cosmos-Reason2-8B experiment. The goal is to determine whether reducing model size attacks the dominant language-generation cost while retaining the ability to ingest a short ordered frame sequence in one inference.

This is primarily a latency characterization. A no-change sanity test is included, but it is not a substitute for a controlled temporal-quality benchmark.

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
| Benchmark output limit | 32 tokens |
| Warmup | 1 iteration per condition |
| Measured iterations | 5 per condition |
| Frame conditions | F1, F2, F4, F8 |

The managed 2B engine was validated as ready before the run and the persistent server was restarted after profile activation.

## F8 capacity smoke test

One eight-frame IPC request succeeded:

```text
success: true
frame_count: 8
inference_seconds: 0.440211
client_latency_ms: 467
```

This establishes F8 capacity for the tested 2B profile.

## Steady-state IPC latency

All 20 measured inferences succeeded.

| Condition | Frames | Mean IPC latency | p95 | Relative to F1 |
| --- | ---: | ---: | ---: | ---: |
| F1 | 1 | 266.2 ms | 280.0 ms | 1.000x |
| F2 | 2 | 249.8 ms | 255.0 ms | 0.938x |
| F4 | 4 | 322.2 ms | 331.0 ms | 1.210x |
| F8 | 8 | 415.4 ms | 422.0 ms | 1.560x |

The small F2-vs-F1 inversion should be treated as run variance/output-policy effects rather than evidence that two frames are intrinsically cheaper than one.

Measured IPC samples:

```text
F1: 276, 280, 259, 262, 254 ms
F2: 250, 254, 255, 240, 250 ms
F4: 329, 331, 305, 319, 327 ms
F8: 419, 422, 404, 420, 412 ms
```

## Native profiled stage timing

| Condition | Frames | Visual tokens | Vision encoder | Prefill | Generated tokens | Generation tok/s | Generation GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 1 | 192 | 16.1 ms | 14.6 ms | 32 | 147.0 | 217.8 ms |
| F2 | 2 | 384 | 24.7 ms | 15.9 ms | 32 | 146.7 | 218.1 ms |
| F4 | 4 | 768 | 46.1 ms | 15.8 ms | 32 | 144.0 | 222.2 ms |
| F8 | 8 | 1536 | 72.8 ms | 24.9 ms | 32 | 138.1 | 231.8 ms |

All direct-profile runs reached the configured 32-token maximum, so generation length is controlled consistently within this comparison.

At F8, exposed stages sum to roughly:

```text
vision encoder:   72.8 ms
prefill:          24.9 ms
generation GPU:  231.8 ms
-------------------------
exposed sum:     329.5 ms
```

Generation remains the largest exposed stage.

## Cosmos-Reason2-8B vs 2B

### Steady-state IPC

| Condition | 8B mean IPC | 2B mean IPC | 2B reduction |
| --- | ---: | ---: | ---: |
| F1 | 708.6 ms | 266.2 ms | 62.4% |
| F2 | 753.8 ms | 249.8 ms | 66.9% |
| F4 | 822.0 ms | 322.2 ms | 60.8% |
| F8 | 957.6 ms | 415.4 ms | 56.6% |

### F8 profiled stages

| Stage | 8B F8 | 2B F8 | Reduction |
| --- | ---: | ---: | ---: |
| Vision encoder | 102.8 ms | 72.8 ms | 29.2% |
| Prefill | 87.3 ms | 24.9 ms | 71.5% |
| Generation GPU | 683.8 ms | 231.8 ms | 66.1% |
| Sum of exposed stages | 873.9 ms | 329.5 ms | 62.3% |

Generation throughput rises from about 45.3 tokens/s on 8B F8 to 138.1 tokens/s on 2B F8. This supports model/decoder size as a higher-leverage latency control than aggressively reducing a short visual history.

## No-change sanity test

The latency fixture contains effectively identical red-panda frames. A stricter F8 prompt asked for a structured change decision. The model correctly returned `change_detected=false`, but also referenced nonexistent `Image 9` and `Image 10` despite receiving only eight frames.

Interpretation:

- coarse no-change detection passed;
- compact structured output was machine-usable;
- exact frame-reference grounding did not pass;
- one earlier unconstrained sample also produced an unrelated object hallucination.

This supports initial viability but does not establish 2B quality equivalence to 8B.

## Conclusion

Eight ordered frames can be processed in one Cosmos-Reason2-2B inference on Thor at roughly **415 ms mean steady-state IPC latency** under the tested 32-token policy, versus roughly **958 ms** for Cosmos-Reason2-8B using the same managed profile limits.

The native profile confirms that much of the gain comes from reducing the dominant language-model cost: F8 generation GPU time falls from about **684 ms to 232 ms**.

Cosmos-Reason2-2B is therefore a strong temporal VLM candidate, with the important caveat that quality must be evaluated on controlled sequences containing real state changes.

## Recommended controlled quality benchmark

Use sequence classes such as:

- object appears / disappears;
- person or vehicle approaches / recedes;
- path becomes blocked / unblocked;
- scene or weather visibility changes;
- other state changes that cannot be answered from the terminal frame alone.

Use chronological, reversed, shuffled, repeated-static, and single-terminal-frame controls. Compare models on change accuracy, direction/trend correctness, event timing, structured-output validity, hallucination rate, calibration, and steady-state latency.
