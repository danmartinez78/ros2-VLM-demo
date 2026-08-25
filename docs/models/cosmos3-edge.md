# Cosmos3-Edge on Jetson Thor: Validation Findings

**Status:** TensorRT Edge-LLM text and single-image paths operational after correcting a Cosmos3 patch-layout mismatch; 1-3 independent images validated; 4-image positional binding remains incorrect; FlashRT BF16 single-image path also validated.  
**Last updated:** 2026-08-24  
**Target:** `nvidia/Cosmos3-Edge` reasoner / understanding checkpoint on Jetson AGX Thor.

This document records the actual hardware bring-up, failed hypotheses, corrected conclusions,
and current runtime decision state for Cosmos3-Edge on Thor.

---

## 1. Executive summary

The original TensorRT Edge-LLM correctness diagnosis was partly wrong in two separate ways.

First, early text-only Edge-LLM tests produced a deterministic `Guanaco` answer. That looked
like a lower-level Cosmos3 model/runtime translation failure, but cross-runtime controls later
showed that the behavior was caused by running Cosmos3 with **thinking disabled**. The exported
Edge-LLM chat template contains both a pre-closed `<think></think>` generation prompt and the
official reasoning prompt ending in `<think>\n`. When the exact malformed 45-token prompt was
fed to raw Hugging Face, Edge-LLM, and FlashRT, all three generated `Guanaco<|im_end|>`.
Therefore the earlier result was prompt-induced rather than evidence of a broken TensorRT text
path.

With:

```json
"enable_thinking": true
```

Edge-LLM selects the official Cosmos3 reasoning suffix and produces a normal reasoning trace.
The Edge-LLM text decoder is therefore no longer considered fundamentally broken.

Second, the original Edge-LLM visual path produced a deterministic corrupted interpretation of
a known red-panda image: black-and-white, abstract, distorted, and grid-like. That failure was
reproduced with both the original Thor/AArch64 export and a supported x86-64 export rebuilt on
Thor.

The visual failure has now been isolated to a **within-patch tensor layout mismatch** between the
Cosmos3 Hugging Face processor / trained SigLIP2 patch embedding and the inherited Qwen-family
Edge-LLM C++ patch packing path:

```text
Cosmos3 HF processor / trained patch embedding:
    16 x 16 patch flattened as H, W, C

Edge-LLM Qwen-style CUDA preprocessing:
    16 x 16 patch flattened as C, H, W
```

A direct numerical reconstruction showed that the original HF `pixel_values` tensor and the
Edge-style packed tensor differed substantially, but converting only the within-patch layout
from CHW to HWC made the tensors **bit-identical**:

```text
shape:          (768, 768) == (768, 768)
max abs diff:   0.0
mean abs diff:  0.0
exact equal:    True
allclose:       True
cosine:         1.0000266
```

The next decisive check compared the original checkpoint patch-embedding weight with the
x86-exported ONNX initializer:

```text
checkpoint:
  model.visual.embeddings.patch_embedding.weight  (1152, 768)

ONNX:
  visual.embeddings.patch_embedding.weight        (1152, 768)
```

The ONNX weight remained effectively identical to the original checkpoint weight:

```text
max abs diff:   2.98e-08
mean abs diff:  1.78e-11
allclose:       True
```

Therefore the export did **not** compensate for the runtime's CHW patch layout. The runtime was
effectively evaluating the original HWC-trained linear weight against CHW-flattened patch
vectors.

A diagnostic copy of the visual ONNX was created with only the patch-embedding input columns
permuted so that:

```text
W_patched @ x_CHW == W_original @ x_HWC
```

Specifically:

```python
patched = (
    original
    .reshape(1152, 16, 16, 3)
    .transpose(0, 3, 1, 2)
    .reshape(1152, 768)
)
```

A new visual TensorRT engine built from that diagnostic ONNX immediately restored correct image
understanding. The same red-panda fixture was correctly described with both thinking enabled
and disabled, and NVIDIA's official reasoning sample was also correctly described as a robotic
arm interacting with objects on a table.

The **single-image Cosmos3 Edge-LLM path on Thor is therefore operational** with the diagnostic
HWC/CHW compensation applied.

A second, independent issue remains open: independent multi-image requests work correctly with
1, 2, and 3 images, but a 4-image request exhibits incorrect image-to-position binding. Engine
capacity is not the cause: the visual engine supports up to 4096 raw patch rows and derives a
`maxNumImages` capacity of 1024 from `cu_seqlens max=(1025,)`; four 512x384 images require only
3072 raw patches / 768 merged image tokens. This issue should be treated separately from the
solved single-image layout bug.

FlashRT BF16 also passed the known red-panda single-image correctness gate. It remains useful as
an independent runtime control and possible alternative backend, but TensorRT Edge-LLM should no
longer be discarded on the basis of the earlier single-image failure.

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
| GPU | NVIDIA Thor, SM110 |

### Checkpoint

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/hf_checkpoint
```

### Known image fixture

```text
/tmp/vlm_multiframe_512/frame_001.jpg
SHA256: c221acdc6eef46309207dfa33c79708ca70b05b51e770375661308d3e6595acb
```

The image contains a red panda on a wooden platform with green foliage.

The fixture dimensions are:

```text
512 x 384 RGB JPEG
```

For Cosmos3 with `patch_size=16` and `merge_size=2`:

```text
patch grid:          32 x 24
raw patches/image:   768
merged tokens/image: 192
```

---

## 3. Raw Hugging Face reference

Raw Transformers on Thor remains the reference implementation for standalone correctness.

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

The official processor output for the red-panda image was:

```text
prompt tokens:       219
image token count:   192
pixel_values:        (768, 768)
image_grid_thw:      [[1, 24, 32]]
```

This establishes that the checkpoint, fixture, and Thor hardware are capable of correct
multimodal inference.

---

## 4. The `Guanaco` investigation and corrected text diagnosis

The diagnostic text prompt was:

```text
What animal is known for reddish-brown fur, white facial markings, and a long
ringed tail? Reply with only the animal name.
```

The original Edge-LLM request omitted `enable_thinking`. The formatted prompt ended with:

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

The exact malformed prompt contained 45 IDs and ended with the tokens corresponding to:

```text
<think></think>
```

Feeding those exact 45 IDs to raw Hugging Face produced the same output:

```text
Guanaco<|im_end|>
```

FlashRT BF16 also produced:

```text
Guanaco<|im_end|>
```

Therefore the repeated answer was caused by the prompt representation, not a shared runtime
corruption.

### 4.2 Official processor difference

Using `AutoProcessor.apply_chat_template(..., add_generation_prompt=True)` also produced 45
tokens, but the final prompt representation ended in:

```text
<think>\n
```

rather than the pre-closed:

```text
<think></think>
```

### 4.3 Edge-LLM thinking mode

The x86-derived Edge-LLM engine's `processed_chat_template.json` contains both:

```text
generation_prompt:          <|im_start|>assistant\n<think></think>
generation_prompt_thinking: <|im_start|>assistant\n<think>\n
```

A request with:

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

This rehabilitates the Edge-LLM **text-only** path. The earlier claim that the Cosmos3 text
decoder was broadly broken is no longer supported.

---

## 5. FlashRT BF16 findings

FlashRT's Thor Cosmos3 path was compiled and loaded successfully in an isolated Docker image.

### 5.1 Text behavior

With the same malformed pre-closed `<think></think>` prompt used in the original Edge-LLM test,
FlashRT produced the same deterministic `Guanaco` output as raw HF and Edge-LLM.

With the official thinking prompt, FlashRT generated a normal reasoning trace ending in:

```text
</think>
Red fox<|im_end|>
```

Exact-prefix recomputation showed that the apparent long-generation divergence between FlashRT
and raw HF was not evidence of gross model misinterpretation; at the same full prefix the two
implementations selected the same top-1 token.

### 5.2 Single-image correctness

FlashRT BF16 was then run on the exact red-panda fixture using the official Cosmos3 processor.
It correctly described:

```text
red panda
wooden structure / platform
green foliage
```

The run completed with approximately:

```text
prompt_tokens: 219
prefill_s:     0.3303
decode_s:      3.6080
decode_tok_s:  70.68
new_tokens:     256
```

Therefore FlashRT BF16 passes the same single-image semantic control as raw HF and the patched
Edge-LLM path.

Temporal / video correctness remains a separate gate.

---

## 6. TensorRT Edge-LLM visual root cause

### 6.1 Original failure

Input:

```text
image: /tmp/vlm_multiframe_512/frame_001.jpg
prompt: Describe the scene in the image.
```

Both the original Thor/AArch64-derived visual engine and a fresh visual engine built on Thor from
the supported x86-exported visual ONNX produced an interpretation resembling:

```text
black and white
abstract
distorted
grid-like / wavy texture
indistinct central shapes
```

Disabling image resize did not change the result, ruling out the resize stage.

### 6.2 Geometry and normalization checks

The known image is already 384x512 after orientation interpretation, and both HF and Edge-LLM
resolve the same geometry:

```text
resized dimensions: 384 x 512
patch grid:          24 x 32
raw patches:         768
merged image tokens: 192
```

The checkpoint and Edge visual configs agree on:

```text
patch_size:          16
merge_size:          2
temporal_patch_size: 1
image_mean:          [0.5, 0.5, 0.5]
image_std:           [0.5, 0.5, 0.5]
```

The Edge-LLM CUDA normalization kernel performs the expected:

```text
(pixel / 255 - mean) / std
```

so normalization itself is not the problem.

### 6.3 HWC versus CHW within each patch

The Cosmos3 Hugging Face processor flattens each patch with **HWC values inside the patch**.
The inherited Edge-LLM Qwen-family `transposeToPatchQwenKernel` flattens each patch in
**CHW order**.

A direct reconstruction of Edge-LLM's packing produced the same `(768, 768)` tensor shape as HF,
but with a large value-order mismatch:

```text
max abs diff:   1.984375
mean abs diff:  0.2163086
exact equal:    False
allclose:       False
cosine:         0.7526678
```

Converting only the reconstructed Edge tensor from CHW to HWC inside each 16x16 patch produced:

```text
max abs diff:   0.0
mean abs diff:  0.0
exact equal:    True
allclose:       True
cosine:         1.0000266
```

This localizes the preprocessing mismatch exactly to the within-patch element ordering.

### 6.4 Export does not compensate

The original reasoning vision checkpoint stores:

```text
model.visual.embeddings.patch_embedding.weight (1152, 768)
model.visual.embeddings.patch_embedding.bias   (1152,)
```

The x86-exported ONNX contains:

```text
visual.embeddings.patch_embedding.weight (1152, 768)
visual.embeddings.patch_embedding.bias   (1152,)
```

The exported ONNX weight is effectively unchanged from the checkpoint:

```text
max abs diff:   2.9802322e-08
mean abs diff:  1.7750254e-11
allclose:       True
```

The bias is an exact match.

Therefore no export-time weight permutation compensates for the C++ runtime's CHW patch layout.

### 6.5 Diagnostic compensation

A diagnostic copy of the visual ONNX was created. Only the patch-embedding weight columns were
permuted:

```python
patched = (
    original
    .reshape(1152, 16, 16, 3)
    .transpose(0, 3, 1, 2)
    .reshape(1152, 768)
)
```

The diagnostic export was verified after serialization:

```text
saved shape:                  (1152, 768)
matches intended permutation: True
still matches original:       False
```

A new visual engine was built successfully from this diagnostic ONNX.

Diagnostic ONNX tree:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx-hwcfix/visual
```

Working diagnostic multimodal engine:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning-mm-hwcfix
```

### 6.6 End-to-end proof

With `enable_thinking=true`, the patched engine correctly described the red panda, including its
reddish-brown fur, white facial markings, wooden platform, and green foliage.

With thinking disabled, the patched engine again correctly described the red panda.

The same engine also correctly described NVIDIA's official reasoning sample as a robotic arm /
gripper positioned above a wooden table with multiple objects.

This is the end-to-end proof that the single-image semantic failure was caused by the patch-layout
contract mismatch and that compensating for it restores correct visual inference.

---

## 7. Multi-image validation

Multi-image tests were performed only after the single-image visual path was fixed.

The fixture sequence used for the initial tests contained alternating scenes:

```text
frame_001: red panda
frame_002: woman + golden retriever on beach
frame_003: red panda
frame_004: woman + golden retriever on beach
```

### 7.1 One image

Pass. Red panda correctly identified.

### 7.2 Two images

Pass. The model correctly distinguished:

```text
Frame 1: red panda
Frame 2: woman + golden retriever on beach
```

and described the scene change.

### 7.3 Three images

Pass. The model correctly bound:

```text
Frame 1: red panda
Frame 2: woman + golden retriever on beach
Frame 3: red panda
```

### 7.4 Four images

The first free-form four-image test incorrectly described all four frames as the red-panda scene.

A stronger binding test replaced the fourth frame with NVIDIA's clearly different robot-arm
sample and constrained the output labels to:

```text
PANDA
BEACH_DOG
ROBOT_ARM
OTHER
```

Expected:

```text
Frame 1: PANDA
Frame 2: BEACH_DOG
Frame 3: PANDA
Frame 4: ROBOT_ARM
```

Observed:

```text
Frame 1: BEACH_DOG
Frame 2: ROBOT_ARM
Frame 3: ROBOT_ARM
Frame 4: OTHER
```

This demonstrates a **multi-image image-to-position binding problem**, not merely a weak
free-form answer.

### 7.5 Capacity checks

The issue is not explained by obvious engine limits.

The working visual engine reports:

```text
input:
  min=(16, 768)
  opt=(2056, 768)
  max=(4096, 768)

cu_seqlens:
  min=(2,)
  opt=(257,)
  max=(1025,)
```

Edge-LLM derives:

```text
maxNumImages = cu_seqlens MAX - 1 = 1024
```

For four 512x384 frames:

```text
raw patches:          4 * 768 = 3072 < 4096
merged visual tokens: 4 * 192 = 768
```

Therefore the 4-image failure is not a simple 4-image capacity threshold or total visual-token
overflow.

The likely remaining surface is the mapping between multiple visual spans / embeddings and their
corresponding language-model image placeholders. This should be treated as a separate Edge-LLM
multi-image issue and does not invalidate the single-image HWC/CHW root cause.

---

## 8. Current correctness matrix

| Test | Raw HF | Edge-LLM original | Edge-LLM HWC/CHW diagnostic fix | FlashRT BF16 |
|---|---|---|---|---|
| Malformed pre-closed thinking prompt | `Guanaco` | `Guanaco` | `Guanaco` | `Guanaco` |
| Official thinking text prompt | normal reasoning | normal reasoning | normal reasoning | normal reasoning |
| Red-panda single image | **red panda** | corrupted abstract visual interpretation | **red panda** | **red panda** |
| NVIDIA official reasoning image | not used as primary control | incorrect path not pursued | **robotic arm scene** | not yet used |
| 2 independent images | not tested | not useful before vision fix | **pass** | not tested |
| 3 independent images | not tested | not useful before vision fix | **pass** | not tested |
| 4 independent images | not tested | not useful before vision fix | **binding incorrect** | not tested |
| Native video / temporal input | not yet validated | not yet validated | **pending** | pending |

---

## 9. What is now ruled out vs. still open for Edge-LLM

### Ruled out or strongly reduced

The current evidence argues against the following as the primary Cosmos3 Edge-LLM problem:

- generic text decoder failure;
- `Guanaco` being a TensorRT-specific semantic failure;
- corrupt red-panda fixture;
- Thor being unable to run Cosmos3 correctly;
- image resize being the source of the distorted visual output;
- image geometry / token-count disagreement between HF and Edge-LLM;
- normalization mismatch;
- Thor/AArch64 ONNX export being the sole cause of the visual failure;
- x86 export alone fixing the visual failure;
- a four-image engine-capacity limit causing the current multi-image binding problem.

### Solved single-image failure

The single-image visual corruption is explained by the Cosmos3 HWC-trained patch representation
being fed through the inherited Qwen-style Edge-LLM CHW patch packing path without export-time
weight compensation.

The isolated weight-column permutation provides an end-to-end semantic fix.

### Still open

The following remain open:

- the proper upstream fix for Cosmos3 patch packing: Cosmos3-specific HWC runtime packing versus
  export-time weight permutation;
- independent 4-image placeholder / visual-span binding;
- native video input correctness;
- temporal reasoning quality;
- comparative Edge-LLM versus FlashRT latency and memory behavior;
- production integration into the ROS-facing runtime abstraction.

---

## 10. Direct-builder diagnostic findings

The experimental checkpoint-direct Edge-LLM builder remains useful historical evidence but is
not the preferred deployment path.

Two independent artifact issues were found:

1. RoPE metadata was emitted under a Transformers-v5-style nested
   `text_config.rope_parameters` shape while the pinned C++ runtime expected normalized fields.
2. The direct-builder `processed_chat_template.json` had an empty `content_types` map, causing
   image requests to fail until known-good image/video sentinel definitions were copied into an
   isolated diagnostic engine directory.

These artifacts were patched only in diagnostic copies. The originals were preserved.

The direct-builder semantic results gathered before the thinking-mode correction should not be
used as proof of a text-decoder defect.

---

## 11. Supported x86 export workflow

The supported workflow used for the current baseline is:

```text
x86-64 Linux host:
  Cosmos3 reasoner ONNX export

Jetson Thor:
  visual / LLM TensorRT engine build
  runtime inference
```

The x86 export was performed on `section9` from the same checkpoint copied from Thor and the same
pinned Edge-LLM revision.

Export environment:

```text
torch:        2.13.0+cu130
torchvision:  0.28.0+cu130
transformers: 5.14.1
cuda_available: False
```

Baseline export:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx
```

Diagnostic HWC/CHW-compensated visual export:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx-hwcfix
```

The baseline export should remain preserved for A/B comparison.

---

## 12. Preserved artifacts on Thor

```text
Checkpoint:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/hf_checkpoint

Original Thor/AArch64-derived engine:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/nvidia-reference-reasoning

Direct-builder original:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/direct-reference

Direct-builder diagnostic copy:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/direct-reference-ropefix

Supported x86-exported ONNX baseline:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx

Diagnostic HWC/CHW-compensated ONNX:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx-hwcfix

x86-derived LLM engine:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning

x86-derived LLM + original visual engine:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning-mm

Working diagnostic multimodal engine:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning-mm-hwcfix

FlashRT checkout:
~/flashrt-cosmos3

FlashRT Docker image:
flashrt:cosmos3-thor

Known red-panda fixture:
/tmp/vlm_multiframe_512/frame_001.jpg

Thinking-enabled successful Edge-LLM output:
/tmp/cosmos3_x86_mm_image_thinking_hwcfix_output.json

Thinking-disabled successful Edge-LLM output:
/tmp/cosmos3_x86_mm_image_no_thinking_hwcfix_output.json

Official NVIDIA sample successful output:
/tmp/cosmos3_official_image_test_output.json

Two-image test output:
/tmp/cosmos3_edge_2frame_test_output.json

Three-image test output:
/tmp/cosmos3_edge_3frame_test_output.json

Four-image binding test output:
/tmp/cosmos3_edge_4frame_binding_test_output.json
```

`/tmp` is not durable project storage; fixtures and results required for permanent reproduction
should eventually be copied into the repository or benchmark artifact store.

---

## 13. Runtime strategy and next gates

The project does not require every model to use the same backend.

The already validated Cosmos-Reason2 Edge-LLM path should remain untouched while Cosmos3 is
evaluated independently.

The current priority is no longer to prove that Cosmos3 can perform image inference on Thor;
that gate is passed in both the patched Edge-LLM path and FlashRT BF16.

Recommended next sequence:

1. Preserve and document the HWC/CHW diagnostic fix and the 4-image binding limitation.
2. Test Cosmos3 through the intended **native video / temporal input path** rather than continuing
   to debug arbitrary four-image placeholders immediately.
3. Establish native temporal correctness on short, controlled sequences.
4. Run chronological / reverse / shuffle / duplicate / terminal-only controls.
5. Benchmark latency only after temporal correctness is established.
6. Compare Edge-LLM and FlashRT as runtime candidates for the ROS-facing abstraction.
7. Return to the ODD temporal reasoning experiments only after the chosen backend passes the
   correctness gates.

The 4-image binding RCA can proceed separately if arbitrary independent multi-image requests
become necessary for the production interface.

---

## 14. Decision record

As of 2026-08-24:

- **Cosmos3 checkpoint:** viable; raw HF text and image controls execute correctly.
- **Edge-LLM text path:** operational. Thinking must be explicitly enabled when the reasoning
  prompt is desired.
- **`Guanaco` result:** explained by the pre-closed `<think></think>` prompt and reproduced in raw
  HF and FlashRT; not a TensorRT-specific failure.
- **Edge-LLM original single-image visual path:** root cause isolated to a within-patch HWC/CHW
  layout mismatch between Cosmos3's trained visual representation and inherited Qwen-style C++
  packing, without export-time patch-embedding compensation.
- **Edge-LLM diagnostic HWC/CHW fix:** successful end-to-end. Red panda and NVIDIA official sample
  are recognized correctly on Thor.
- **Edge-LLM independent multi-image path:** 1, 2, and 3 images validated; 4-image positional
  binding is incorrect despite ample engine capacity.
- **FlashRT BF16:** text and known red-panda single-image paths operational.
- **Performance benchmarking:** still deferred until native temporal / video correctness is
  established.
- **Next action:** test Cosmos3 through its native temporal / video interface using the preserved
  working visual artifacts, while leaving the 4-image placeholder-binding issue documented as a
  separate open defect.
