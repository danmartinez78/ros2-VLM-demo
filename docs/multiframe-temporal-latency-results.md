# Thor multi-frame VLM latency characterization

Date: 2026-08-23 (America/Chicago; reports generated 2026-08-24 UTC)

## Research question

Can useful short-horizon temporal context be supplied to a VLM as multiple ordered
frames in **one inference** at only modest marginal latency, compared with invoking
the VLM independently for each frame?

This experiment characterizes the latency side of that question on Jetson AGX
Thor. It does **not** by itself establish temporal-reasoning quality; task-level
quality must be evaluated separately on sequences that require motion/change
reasoning.

## Tested configuration

| Item | Configuration |
| --- | --- |
| Hardware | NVIDIA Jetson AGX Thor |
| JetPack / L4T | JetPack 7.2 / R39.2.x |
| Model | `nvidia/Cosmos-Reason2-8B` |
| Quantization | NVFP4 |
| TensorRT Edge-LLM | pinned repository build used by this project |
| Runtime mode | persistent IPC server for steady-state latency; native `llm_inference --dumpProfile` for stage timing |
| Prompt policy | one compact temporal JSON result for the full ordered frame set |
| Max output tokens | 32 |
| Warmup | 1 iteration per condition |
| Measured iterations | 5 per condition |
| Frame conditions | F1, F2, F4, F8 |

The final comparison uses one common managed engine profile, `thor-f8`, for all
four frame-count conditions:

```text
maxBatchSize:             1
maxInputLen:              2048
maxKVCacheCapacity:       4096
maxImageTokens:           2048
maxImageTokensPerImage:    512
decode.strategy:          standard
```

The managed engine is stored separately from the original control engine:

```text
~/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engine/             # legacy control
~/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engines/thor-f8/   # managed F8 engine
```

The visual engines were confirmed to be distinct artifacts. On the Thor used for
this experiment:

```text
legacy visual.engine SHA-256:
997f20c9f7614e038235df3728682f2e84503a8788452d5a5f7d93ac4fbedcc5

thor-f8 visual.engine SHA-256:
79b1f344e188672e8390141ccfc194b1a0de39dbafd872e4f567d5f218d6b98a
```

The legacy visual config used `max_image_tokens=1024`; `thor-f8` used
`max_image_tokens=2048`. Both used `max_image_tokens_per_image=512`.

## Steady-state IPC results

The persistent `edge_vlm_server` was loaded with the activated `thor-f8` engine.
All 20 measured inferences succeeded, and measured outputs were held to the same
12-word response length.

| Condition | Frames | Mean IPC latency | p95 IPC latency | Increase vs F1 | Relative to F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | 1 | 708.6 ms | 730.0 ms | - | 1.000x |
| F2 | 2 | 753.8 ms | 759.0 ms | +45.2 ms | 1.064x |
| F4 | 4 | 822.0 ms | 833.0 ms | +113.4 ms | 1.160x |
| F8 | 8 | 957.6 ms | 970.0 ms | +249.0 ms | 1.351x |

The key result is that visual temporal context increased by **8x** from F1 to F8
while steady-state request latency increased by only **35.1%**.

For comparison, repeating the measured F1 request sequentially would cost
approximately:

| Target context | Repeated F1 latency | One multi-frame request | Reduction |
| --- | ---: | ---: | ---: |
| 2 frames | 1417.2 ms | 753.8 ms | 46.8% |
| 4 frames | 2834.4 ms | 822.0 ms | 71.0% |
| 8 frames | 5668.8 ms | 957.6 ms | 83.1% |

These are latency comparisons only; repeated independent inference and one joint
multi-frame inference are not semantically identical algorithms.

## Native profiled results

The same `thor-f8` engine was then benchmarked through NVIDIA
`llm_inference --dumpProfile`, which exposes visual-token count and authoritative
runtime stage timing.

| Condition | Frames | Visual tokens | Vision encoder | Prefill | Generated tokens | Generation tok/s | Generation GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 1 | 192 | 22.0 ms | 33.3 ms | 31 | 46.8 | 662.4 ms |
| F2 | 2 | 384 | 33.9 ms | 34.1 ms | 31 | 46.7 | 663.2 ms |
| F4 | 4 | 768 | 58.3 ms | 49.3 ms | 31 | 46.2 | 670.6 ms |
| F8 | 8 | 1536 | 102.8 ms | 87.3 ms | 31 | 45.3 | 683.8 ms |

From F1 to F8, the incremental profiled cost was:

```text
vision encoder:    +80.8 ms
prefill:           +54.0 ms
generation GPU:    +21.4 ms
--------------------------------
profiled-stage sum +156.2 ms
```

The sum of the three exposed stages increased from approximately 717.7 ms at F1
to 873.9 ms at F8. Even at F8, generation remained the dominant exposed cost
(~684 ms), while generation throughput decreased only modestly from 46.8 to
45.3 tokens/s.

This is important for optimization direction: increasing temporal visual context
mainly increases vision encoding and prefill, while the large-model language
generation cost remains comparatively fixed.

## F8 engine-capacity validation

The original control visual engine could not process the F8 request. It failed
with:

```text
Visual token count 4608 exceeds the ViT engine budget maxHW = 4096
(for video this grows with frame count; rebuild the visual engine with a larger
--maxImageTokens).
```

The managed `thor-f8` engine was built and validated through `modelctl`, activated,
and successfully handled F8 through both runtime paths.

A direct native F8 run on the correct engine reported:

```text
total_images:            8
total_image_tokens:      1536
vision_encoder:          ~103-114 ms
prefill:                 ~87-92 ms
generation GPU:          ~684-705 ms
generated tokens:        31
```

This confirms that the `maxImageTokens=2048` profile provides sufficient capacity
for the tested eight-frame workload.

## Reproducibility incident: stale engine environment

The first direct F8 benchmark attempt failed with the old `maxHW=4096` limit even
though the output directory name referred to `thor-f8`. Investigation showed:

1. the new visual engine existed and had a different SHA-256 from the legacy engine;
2. its config correctly declared `max_image_tokens=2048`;
3. a native `llm_inference` invocation using hard-coded absolute `thor-f8` engine
   paths succeeded with eight images and 1536 reported visual tokens;
4. rerunning the benchmark after sourcing the activated runtime environment
   succeeded for F1/F2/F4/F8.

The failure was therefore an engine-path/environment reproducibility problem, not
an F8 engine-capacity failure or a multi-image semantic difference between the
native and IPC runtimes.

Before running hardware benchmarks, explicitly source and print the active model
profile and engine paths:

```bash
source scripts/edge_vlm_env.sh

echo "MODEL=$EDGE_VLM_MODEL_NAME"
echo "PROFILE=$EDGE_VLM_ENGINE_PROFILE_ID"
echo "LLM=$EDGE_VLM_LLM_ENGINE_DIR"
echo "MM=$EDGE_VLM_MULTIMODAL_ENGINE_DIR"
```

For this experiment the expected profile was:

```text
PROFILE=thor-f8
LLM=/home/daniel/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engines/thor-f8/llm
MM=/home/daniel/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engines/thor-f8
```

Future benchmark manifests should record the absolute LLM and multimodal engine
paths plus the active engine profile/manifest identity so an output-directory
name cannot be mistaken for proof of the engine actually used.

## Comparison with the legacy control engine

Earlier steady-state measurements used the original control engine and cannot be
mixed directly with F8 results from the new profile. The common conditions show
that changing TensorRT profile limits can move performance slightly:

| Condition | Legacy engine | `thor-f8` engine | Difference |
| --- | ---: | ---: | ---: |
| F1 | 739.4 ms | 708.6 ms | -30.8 ms (-4.2%) |
| F2 | 758.0 ms | 753.8 ms | -4.2 ms (-0.6%) |
| F4 | 801.0 ms | 822.0 ms | +21.0 ms (+2.6%) |

Therefore the definitive F1/F2/F4/F8 scaling curve is the run in which all four
conditions use the same `thor-f8` engine.

## Interpretation

The latency hypothesis is supported strongly on this configuration:

> For short temporal windows on Cosmos-Reason2-8B/Thor, supplying several ordered
> frames to one VLM inference is substantially more latency-efficient than
> repeating independent VLM inference for every frame.

F4 is an attractive low-latency operating point in this test (~822 ms steady
state), while F8 provides twice the temporal context at ~958 ms and remains below
one second mean steady-state latency.

The experiment also shows that reducing model-generation cost is likely a higher
leverage optimization than aggressively reducing frame count. At F8, the vision
and prefill stages grow as expected, but generation is still the largest exposed
latency component.

## What this result does not prove

This benchmark intentionally isolates latency scaling. The fixture/smoke sequence
is sufficient to verify multi-image ingestion and consistent model behavior, but
it is not a rigorous temporal-reasoning quality benchmark.

Do not infer from these numbers alone that F8 is always more accurate or useful
than F4/F2. The next quality experiments should include sequences where the answer
requires temporal evidence, for example:

- object motion or direction;
- appearance/disappearance;
- changing weather/visibility;
- progression toward an ODD boundary;
- transient hazards;
- scene-state changes that cannot be answered reliably from one frame.

## Next experiment

The dominant ~660-684 ms generation term motivates repeating the same F1/F2/F4/F8
matrix with `Cosmos-Reason2-2B` using the same managed-profile workflow.

Primary question:

> Can Cosmos-Reason2-2B retain sufficient temporal reasoning quality at F4/F8
> while materially reducing the dominant language-generation latency?

Keep the same frame fixtures, prompt policy, output-token limit, engine-profile
limits where compatible, warmup/iteration policy, and IPC/native measurement
paths so the 2B and 8B results remain comparable.

## Reproduction commands

Steady-state IPC:

```bash
source scripts/edge_vlm_env.sh

bash scripts/benchmark/run_vlm_multiframe_benchmark.sh \
  --sequence-dir /tmp/vlm_multiframe_512 \
  --frame-counts 1,2,4,8 \
  --max-output-tokens 32 \
  --warmup 1 \
  --iterations 5 \
  --paths ipc \
  --output-dir /tmp/vlm_multiframe_f1248_thor_f8
```

Native profiling:

```bash
source scripts/edge_vlm_env.sh

bash scripts/benchmark/run_vlm_multiframe_benchmark.sh \
  --sequence-dir /tmp/vlm_multiframe_512 \
  --frame-counts 1,2,4,8 \
  --max-output-tokens 32 \
  --warmup 1 \
  --iterations 5 \
  --paths direct \
  --output-dir /tmp/vlm_multiframe_f1248_direct_thor_f8
```

The corresponding report files are generated as
`vlm_multiframe_report.txt` and `vlm_multiframe_report.json` under each output
directory.
