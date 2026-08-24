# Cosmos3-Edge on Jetson Thor: Validation Findings

**Status:** TensorRT Edge-LLM text path rehabilitated with thinking enabled; Edge-LLM multimodal path still incorrect; FlashRT BF16 validation in progress.  
**Last updated:** 2026-08-24  
**Target:** `nvidia/Cosmos3-Edge` reasoner / understanding checkpoint on Jetson AGX Thor.

This document records the actual hardware bring-up, failed hypotheses, corrected conclusions,
and current runtime decision state for Cosmos3-Edge on Thor.

---

## 1. Executive summary

The original TensorRT Edge-LLM correctness diagnosis was partly wrong.

Early Edge-LLM tests produced a deterministic text answer of `Guanaco` and incorrect
image descriptions. That initially looked like a lower-level Cosmos3 model/runtime
translation failure. Later cross-runtime testing showed that the text failure was
caused by running Cosmos3 with **thinking disabled**.

The exported Edge-LLM chat template contains both:

```text
generation_prompt:          <|im_start|>assistant\n<think></think>
generation_prompt_thinking: <|im_start|>assistant\n<think>\n
```

`enable_thinking` defaults to `false` in the Edge-LLM request schema. The original
request JSON omitted the field, so Edge-LLM selected the pre-closed
`<think></think>` generation prompt.

That behavior was reproduced exactly in raw Hugging Face and FlashRT using the same
45 input token IDs: all three paths generated `Guanaco<|im_end|>`. This proves that
`Guanaco` was **prompt-induced**, not evidence that Edge-LLM text inference was
fundamentally broken.

With:

```json
"enable_thinking": true
```

Edge-LLM instead selects the official Cosmos3 reasoning suffix:

```text
<|im_start|>assistant
<think>
```

and produces a normal reasoning trace. On the text clue used during diagnosis,
Edge-LLM converged on `Red Fox`, closely matching the FlashRT BF16 trajectory for the
same prompt. The clue itself is therefore not a useful semantic golden; runtime
parity matters more than whether the model solves that trivia example correctly.

The **multimodal Edge-LLM path remains incorrect**. The known red-panda fixture is
correctly recognized by raw HF, but Edge-LLM describes it as an abstract,
black-and-white, distorted/grid-like image. This failure persists with:

1. the original Thor/AArch64-exported visual engine; and
2. a newly built Thor visual engine from the supported x86-64-exported visual ONNX.

The current Edge-LLM blocker is therefore specifically in the Cosmos3 multimodal /
vision path, not the text decoder in general.

FlashRT is being evaluated independently. Its Thor-specific Cosmos3 BF16 path builds
and runs successfully, and its text behavior is consistent with raw HF when the same
prompt representation is used. The next FlashRT gate is the known red-panda image
using the official processor output.

---

## 2. Revisions and environment

### TensorRT Edge-LLM

| Item | Value |
|---|---|
| Repository | `NVIDIA/TensorRT-Edge-LLM` |
| Pinned commit | `71dd1bae032e70771265917ec74d3ff4cad07a10` |
| Version | `0.10.0` |
| Thor platform | Jetson AGX Thor |
| JetPack / L4T | JetPack 7.2 / L4T R39.2.0 |
| CUDA | 13.2.78 |
| Host TensorRT | 10.16.2.10 |
| Native plugin | `~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so` |

### FlashRT

| Item | Value |
|---|---|
| Repository | `flashrt-project/FlashRT` |
| Pinned commit | `f72192b263b267994edd7bbff0a8c62c6da98948` |
| Thor checkout | `~/flashrt-cosmos3` |
| Docker image | `flashrt:cosmos3-thor` |
| Base image | `nvcr.io/nvidia/pytorch:25.09-py3` |
| GPU | NVIDIA Thor, SM110 |

The FlashRT Docker image was built from an isolated Cosmos3-specific Thor Dockerfile
with:

```text
GPU_ARCH=110
FLASHRT_ENABLE_COSMOS3_REASONER=ON
```

The build completed successfully and exported the expected Cosmos3 reasoner kernel
symbols.

### Checkpoint

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/hf_checkpoint
```

### Known image fixture

```text
/tmp/vlm_multiframe_512/frame_001.jpg
SHA256: c221acdc6eef46309207dfa33c79708ca70b05b51e770375661308d3e6595acb
```

The image visibly contains a red panda on a wooden platform with green foliage.

---

## 3. Raw Hugging Face reference

Raw Transformers on Thor remains the reference implementation for standalone
correctness.

Environment used during the successful visual control:

```text
nvcr.io/nvidia/pytorch:26.05-py3
Cosmos3EdgeForConditionalGeneration
AutoProcessor / Cosmos3EdgeProcessor
bfloat16
greedy generation
```

For:

```text
Describe the scene in the image.
```

raw HF correctly identifies and describes the **red panda**.

This establishes that the checkpoint, image fixture, and Thor hardware are capable of
correct multimodal inference.

---

## 4. The `Guanaco` investigation and corrected text diagnosis

The diagnostic text prompt was:

```text
What animal is known for reddish-brown fur, white facial markings, and a long
ringed tail? Reply with only the animal name.
```

The original Edge-LLM request omitted `enable_thinking`. The formatted prompt ended
with:

```text
<|im_start|>assistant
<think></think>
```

and Edge-LLM returned:

```text
Guanaco<|im_end|>
```

This initially appeared to prove a text-model correctness defect.

### 4.1 Exact-token control

The exact malformed prompt tokenization contained 45 IDs and ended with:

```text
12, 13
```

corresponding to:

```text
<think></think>
```

Feeding the **same exact 45 IDs** to raw Hugging Face produced the same output:

```text
Guanaco<|im_end|>
```

FlashRT BF16 also produced:

```text
Guanaco<|im_end|>
```

Therefore the repeated answer was caused by the prompt representation, not a shared
runtime corruption.

### 4.2 Official processor difference

Using `AutoProcessor.apply_chat_template(..., add_generation_prompt=True)` produced
45 tokens as well, but differed at the final token only:

```text
manual / thinking-disabled: ... 12, 13
actual Cosmos3 reasoning:   ... 12, 1010
```

Decoded:

```text
manual:   <think></think>
official: <think>\n
```

The leading newline at the start of the template was initially suspected, but it is
part of the official processor output and is not the issue.

### 4.3 Edge-LLM thinking mode

The x86-derived Edge-LLM engine's `processed_chat_template.json` already contained:

```json
"generation_prompt": "<|im_start|>assistant\n<think></think>",
"generation_prompt_thinking": "<|im_start|>assistant\n<think>\n"
```

Edge-LLM's request format documents `enable_thinking` as a top-level field with
`false` as the default.

A diagnostic request with:

```json
"enable_thinking": true,
"temperature": 0.0,
"max_generate_length": 256
```

produced a coherent reasoning trace ending in:

```text
</think>
Red Fox<|im_end|>
```

This rehabilitates the Edge-LLM **text-only** path. The earlier claim that the
Cosmos3 text decoder was broadly broken is no longer supported.

---

## 5. FlashRT BF16 text findings

FlashRT's Thor Cosmos3 path was compiled and loaded successfully in the isolated
Docker image.

The engine constructor used for diagnostics was:

```python
CosmosReasonerThor(
    "/model",
    quant="bf16",
    max_new_tokens=256,
    use_graph=True,
)
```

### 5.1 Malformed prompt parity

With the same pre-closed `<think></think>` prompt used in the original Edge-LLM test,
FlashRT produced the same deterministic `Guanaco` output as raw HF and Edge-LLM.

### 5.2 Official thinking prompt

With the official 45-token prompt ending in `<think>\n`, FlashRT generated a normal
reasoning trace ending in:

```text
</think>
Red fox<|im_end|>
```

A raw-HF `generate()` run followed the same first 34 generated tokens and then took a
different autoregressive reasoning branch. To isolate whether this represented a
FlashRT model defect, the shared 79-token prefix was recomputed from scratch through
both implementations.

At the first apparent divergence, FlashRT's pure-Torch reference logits were:

```text
'animal'      19.250
'answer'      19.125
'description' 18.750
```

Raw HF recomputation at the exact same prefix returned:

```text
'animal'      19.250
'answer'      19.125
'description' 18.875
```

Both implementations therefore selected the same top-1 token when given the exact
same full prefix. The earlier long-generation branch difference is consistent with
small cached autoregressive numerical differences rather than gross model
misinterpretation.

Turning FlashRT CUDA graph capture off produced the same output trajectory as graph
mode, ruling out graph capture as the source of the branch difference.

Current conclusion: FlashRT BF16 text execution is sufficiently consistent with the
HF reference to proceed to the multimodal correctness gate.

---

## 6. TensorRT Edge-LLM multimodal failure

The image control remains the decisive Edge-LLM blocker.

Input:

```text
image: /tmp/vlm_multiframe_512/frame_001.jpg
prompt: Describe the scene in the image.
enable_thinking: true
```

### 6.1 Original Thor/AArch64-exported visual engine

With thinking enabled, the model no longer emits a trivial object misclassification.
Instead it reasons about an image that appears to it as:

```text
black and white
abstract
distorted
grid-like / wavy texture
indistinct central shapes
```

The final response likewise describes an abstract distorted monochrome scene.

This is qualitatively more useful than the earlier `dog` answer because it strongly
suggests that the model is receiving malformed visual information rather than merely
choosing the wrong class.

### 6.2 Supported x86-64 visual export and Thor build

The supported x86-64 reasoner export contains a complete visual tree:

```text
visual/config.json
visual/model.onnx
visual/model.onnx.data
visual/preprocessor_config.json
```

A fresh visual engine was built on Thor and combined with the already working
x86-derived LLM engine into:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning-mm
```

Resulting layout:

```text
config.json
embedding.safetensors
llm.engine
processed_chat_template.json
tokenizer_config.json
tokenizer.json
visual/config.json
visual/preprocessor_config.json
visual/visual.engine
```

The same image request with `enable_thinking=true` again produced the abstract,
black-and-white, distorted/grid-like interpretation.

Therefore the Edge-LLM multimodal failure is **not fixed by moving the visual ONNX
export to the documented x86-64 workflow**.

---

## 7. Current correctness matrix

| Test | Raw HF | Edge-LLM Thor export | Edge-LLM x86 export | FlashRT BF16 |
|---|---|---|---|---|
| Malformed pre-closed thinking prompt | `Guanaco` | `Guanaco` | `Guanaco` | `Guanaco` |
| Official thinking text prompt | normal reasoning | normal reasoning | normal reasoning / `Red Fox` | normal reasoning / `Red fox` |
| Known red-panda image | **red panda** | corrupted abstract visual interpretation | corrupted abstract visual interpretation | pending |
| Runtime completion | yes | yes | yes | yes |

The malformed-prompt row is intentionally preserved because it demonstrates the
importance of input equivalence: three independent runtimes agreed when given the
same wrong prompt.

---

## 8. What is now ruled out vs. still open for Edge-LLM

### Ruled out or strongly reduced

The current evidence argues against the following as the primary Cosmos3 Edge-LLM
problem:

- generic text decoder failure;
- the deterministic `Guanaco` result being a TensorRT-specific semantic failure;
- CUDA graph behavior in FlashRT explaining the text discrepancy;
- corrupt image fixture;
- Thor being unable to run Cosmos3 correctly;
- Thor/AArch64 ONNX export being the sole cause of the visual failure.

### Current Edge-LLM failure surface

The remaining failure is concentrated in the Cosmos3 multimodal path, including:

- image preprocessing / normalization / packing;
- visual ONNX or visual-engine runtime contract;
- visual feature projection;
- placement of visual embeddings into the language-model token stream;
- visual mRoPE / geometry metadata;
- a Cosmos3-specific mismatch between the exported visual model and the C++ runtime.

The distorted monochrome/grid-like description is more consistent with bad visual
features than ordinary model uncertainty.

---

## 9. Direct-builder diagnostic findings

The experimental checkpoint-direct Edge-LLM builder remains useful historical
evidence but is not the preferred deployment path.

Two independent artifact issues were found:

1. RoPE metadata was emitted under a Transformers-v5-style nested
   `text_config.rope_parameters` shape while the pinned C++ runtime expected
   normalized fields such as `text_config.rope_theta` and `rope_scaling`.
2. The direct-builder `processed_chat_template.json` had an empty `content_types`
   map, causing image requests to fail until the known-good ONNX-generated
   image/video sentinel definitions were copied into a diagnostic engine directory.

These artifacts were patched only in isolated diagnostic copies. The originals were
preserved.

The direct-builder semantic results gathered before the thinking-mode correction
should not be used as proof of a text-decoder defect.

---

## 10. Supported x86 export workflow

NVIDIA documents the relevant platform split as:

```text
x86-64 Linux host:
  export / optional quantization

Jetson Thor:
  C++ engine build
  runtime inference
```

The x86 export was performed on `section9` from the exact checkpoint copied from
Thor, using the same pinned Edge-LLM revision.

Export environment after resolving the Torch/torchvision mismatch:

```text
torch:        2.13.0+cu130
torchvision:  0.28.0+cu130
transformers: 5.14.1
cuda_available: False
```

The complete export was transferred back to:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx
```

This tree now serves as the supported export baseline for any future Edge-LLM RCA.

---

## 11. Preserved artifacts on Thor

```text
Checkpoint:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/hf_checkpoint

Original Thor/AArch64-derived engine:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/nvidia-reference-reasoning

Direct-builder original:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/direct-reference

Direct-builder diagnostic copy:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/direct-reference-ropefix

Supported x86-exported ONNX tree:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx

x86-derived LLM engine:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning

x86-derived LLM + fresh visual engine:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning-mm

FlashRT checkout:
~/flashrt-cosmos3

FlashRT Docker image:
flashrt:cosmos3-thor

Original text-only request:
/tmp/cosmos3_text_only.json

Thinking-enabled text request:
/tmp/cosmos3_text_only_thinking.json

Original image request:
/tmp/cosmos3_edge_smoke_f1.json

Thinking-enabled image request:
/tmp/cosmos3_edge_smoke_f1_thinking.json

Known image:
/tmp/vlm_multiframe_512/frame_001.jpg
```

`/tmp` is not durable project storage; fixtures/results required for permanent
reproduction should eventually be copied into the repository or benchmark artifact
store.

---

## 12. Runtime strategy and next gates

The project does not require every model to use the same backend.

The already validated Cosmos-Reason2 Edge-LLM path should remain untouched while
Cosmos3 is evaluated independently.

Current sequence:

1. **FlashRT BF16 image correctness** using the exact known red-panda fixture and
   official Cosmos processor output.
2. If image correctness passes, test FlashRT video / multi-frame input.
3. Benchmark BF16 latency only after correctness is established.
4. Evaluate FlashRT FP4 correctness and performance.
5. Integrate a Cosmos3 FlashRT backend behind the existing ROS-facing abstraction.
6. Return to temporal F1/F2/F4/F8 experiments and ODD tasks.

Potential architecture:

```text
ROS2 VLM/reasoning abstraction
  |
  +-- Cosmos-Reason2 -> TensorRT Edge-LLM
  |
  +-- Cosmos3-Edge   -> FlashRT   (if multimodal validation passes)
```

A deeper Edge-LLM visual RCA can proceed separately if repairing that backend becomes
valuable.

---

## 13. Decision record

As of 2026-08-24:

- **Cosmos3 checkpoint:** viable; raw HF text and image controls execute correctly.
- **Edge-LLM text path:** no longer considered fundamentally broken. Thinking must be
  explicitly enabled for Cosmos3 reasoning requests.
- **`Guanaco` result:** explained by the pre-closed `<think></think>` prompt and
  reproduced in raw HF and FlashRT; not a TensorRT-specific failure.
- **Edge-LLM multimodal path:** still blocked. Both Thor/AArch64-derived and supported
  x86-export-derived visual engines produce a corrupted abstract visual
  interpretation of the known red-panda fixture.
- **FlashRT BF16 text path:** operational and sufficiently consistent with exact-prefix
  HF controls to continue validation.
- **Next action:** validate the known red-panda image through FlashRT using official
  processor output before any performance benchmarking or ROS integration.
