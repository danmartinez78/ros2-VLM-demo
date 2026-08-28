# Cosmos3-Nano offline teacher

This directory captures the validated local **Cosmos3-Nano + vLLM** workflow used to generate offline temporal labels on NVIDIA Jetson AGX Thor.

The teacher path is deliberately separate from the live ROS/runtime adapters:

```text
saved temporal capture
        |
        v
prepare_temporal_variants.py
        |
        +-- forward.mp4
        +-- reverse.mp4
        +-- shuffled.mp4
        +-- static_terminal.mp4
        |
        v
Cosmos3-Nano vLLM server
        |
        v
structured JSON teacher results
```

The output should be treated as **silver labels**, not unquestioned ground truth. A human-reviewed subset can later become a gold evaluation set.

## Validated Thor configuration

Validated on Jetson AGX Thor with 128 GB unified memory:

| Component | Configuration |
| --- | --- |
| vLLM image | `vllm/vllm-openai:v0.23.0-aarch64-ubuntu2404` |
| vLLM | 0.23.0 |
| Transformers | 5.12.0 |
| Checkpoint | `nvidia/Cosmos3-Nano` |
| Served model name | `cosmos3-nano-teacher` |
| Max model length | 16384 |
| GPU memory utilization | 0.55 |
| Host media root | `~/cosmos3_teacher` |
| Container media root | `/data` |
| Hugging Face token | `~/.cache/huggingface/token` |

The `0.55` memory-utilization setting was used because the Thor GPU and CPU share the same unified-memory pool. Higher values can leave too little headroom for the host during model loading and initialization.

The first inference after server startup can include substantial warm-up cost. Measure steady-state requests separately from cold start.

## 1. Prepare Thor

The setup script verifies Docker access, installs FFmpeg if required, pulls the validated ARM64 vLLM image, verifies the vLLM/Transformers versions, and confirms that `Cosmos3OmniForConditionalGeneration` is importable.

```bash
cd ~/ros2_ws/src/ros2-VLM-demo
bash teacher/cosmos3_nano/setup_thor.sh
```

The script expects the Hugging Face token at:

```text
~/.cache/huggingface/token
```

Override with `HF_TOKEN_PATH` if necessary.

## 2. Start the teacher server

```bash
cd ~/ros2_ws/src/ros2-VLM-demo
bash teacher/cosmos3_nano/start_vllm_thor.sh
```

The launcher uses the working Thor configuration and mounts:

```text
~/cosmos3_teacher -> /data
```

Useful environment overrides:

```bash
COSMOS3_NANO_PORT=8000
COSMOS3_NANO_MAX_MODEL_LEN=16384
COSMOS3_NANO_GPU_MEMORY_UTILIZATION=0.55
COSMOS3_NANO_DATA_ROOT="$HOME/cosmos3_teacher"
COSMOS3_NANO_MODEL=nvidia/Cosmos3-Nano
COSMOS3_NANO_SERVED_NAME=cosmos3-nano-teacher
```

Do not add `vllm serve` after the container image. The `vllm/vllm-openai` image already uses `ENTRYPOINT ["vllm", "serve"]`.

## 3. Smoke-test the server

In another terminal:

```bash
cd ~/ros2_ws/src/ros2-VLM-demo
python3 teacher/cosmos3_nano/smoke_test.py
```

The smoke test checks `/v1/models`, then performs a deterministic text generation and expects:

```text
COSMOS3 NANO READY
```

## 4. Generate temporal media from a saved capture

The existing FlashRT chronology harness saves frames and timestamps under `temporal_captures/capture_*`. Convert one capture into controlled teacher-model media with:

```bash
cd ~/ros2_ws/src/ros2-VLM-demo

python3 teacher/cosmos3_nano/prepare_temporal_variants.py \
  temporal_captures/capture_20260825T221939Z
```

By default the generated media is written outside the repository under:

```text
~/cosmos3_teacher/generated/<capture-name>/
```

This is intentional: that host directory is already mounted into the vLLM container as `/data`, so the generated files are immediately accessible through local `file:///data/...` URLs.

The generator creates:

- `forward.mp4` — original frame order
- `reverse.mp4` — reversed frame order
- `shuffled.mp4` — deterministic alternating first/last order
- `static_terminal.mp4` — terminal frame repeated for the original frame count
- `terminal_only.png` — final source frame for later single-image controls
- `teacher_media_manifest.json` — source provenance, frame orders, effective FPS, host paths, container paths, and FFprobe results

The constant video frame rate is derived from the saved capture as:

```text
(frame_count - 1) / timestamp_span
```

The source capture timestamps remain recorded in the generated manifest.

## 5. Run one video manually

```bash
python3 teacher/cosmos3_nano/vllm_client.py \
  /data/generated/capture_20260825T221939Z/forward.mp4 \
  --output ~/cosmos3_teacher/generated/capture_20260825T221939Z/forward_result.json
```

The client uses the versioned `temporal_observation_v1.schema.json` response schema and vLLM structured output so the result is machine-readable rather than unconstrained prose.

## 6. Run the controlled temporal suite

```bash
python3 teacher/cosmos3_nano/run_temporal_suite.py \
  ~/cosmos3_teacher/generated/capture_20260825T221939Z/teacher_media_manifest.json
```

By default it evaluates:

```text
forward
reverse
shuffled
static_terminal
```

and writes:

```text
~/cosmos3_teacher/generated/<capture-name>/cosmos3_nano_results.json
```

Each result retains:

- variant name and frame order
- generated-media probe metadata
- request model/media provenance
- wall-clock request latency
- parsed structured teacher output
- complete raw OpenAI-compatible vLLM response

Run a subset with:

```bash
python3 teacher/cosmos3_nano/run_temporal_suite.py \
  ~/cosmos3_teacher/generated/capture_20260825T221939Z/teacher_media_manifest.json \
  --variants forward,reverse
```

## Current schema status

`temporal_observation_v1.schema.json` intentionally captures the first working structured-output contract rather than pretending the task ontology is settled. It currently records:

- objects and observed state
- temporal change category
- direction
- evidence
- camera motion
- temporal coherence
- unsupported claims
- overall confidence

Schema changes should create a new version instead of silently changing historical benchmark semantics.

## Relationship to the rest of the repository

This path does **not** replace or modify:

- TensorRT Edge-LLM inference workers
- FlashRT/Cosmos3 Edge temporal inference
- ROS `InferenceRequest` or `VlmResult`
- the versioned IPC protocol

It is an offline model-agnostic teacher/evaluation utility. The vLLM endpoint can later be replaced by another OpenAI-compatible teacher service without changing the live ROS pipeline.
