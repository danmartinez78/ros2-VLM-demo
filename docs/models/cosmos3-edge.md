# Cosmos3-Edge on Jetson Thor: Validation Findings

**Status:** TensorRT Edge-LLM path blocked on correctness; FlashRT evaluation next.  
**Last updated:** 2026-08-24  
**Target:** `nvidia/Cosmos3-Edge` reasoner / understanding checkpoint on Jetson AGX Thor.

This document replaces the original Phase 1 paper audit with the results of actual
hardware bring-up and correctness testing. The goal is to preserve what was tried,
what was learned, and which conclusions are justified before the project moves to a
second runtime implementation.

---

## 1. Executive summary

`nvidia/Cosmos3-Edge` runs correctly through the raw Hugging Face / Transformers
checkpoint path on Thor, but the same checkpoint does **not** currently produce
trustworthy semantic output through TensorRT Edge-LLM 0.10.0 in this environment.

The failure is broader than the visual path:

| Path | Text-only control | Image control |
|---|---|---|
| Raw HF / Transformers | semantically reaches **red panda** | **red panda** |
| TRT Edge-LLM, ONNX exported on Thor/AArch64 | **Guanaco** | **dog** |
| TRT Edge-LLM, ONNX exported on supported x86-64 host | **Guanaco** | not rebuilt; text gate already failed |
| TRT Edge-LLM, experimental direct builder | **Guanaco** | **dog** |

The supported x86 export test is especially important. NVIDIA documents export and
quantization as an **x86-64 host** step and the C++ engine build/runtime as a target
hardware step. The project's first ONNX export was performed on Thor/AArch64, which
was outside that documented workflow. A clean x86-64 re-export was therefore done
using the same checkpoint and pinned Edge-LLM revision, transferred back to Thor,
and rebuilt natively. The text-only result remained exactly `Guanaco`.

Therefore:

- The original AArch64 export was unsupported, but it was **not the root cause** of
  the observed text correctness failure.
- The failure is **not limited to vision preprocessing**.
- The failure is **not ONNX-only**, because the experimental direct builder also
  produced the same wrong text answer after its artifact issues were worked around.
- The raw checkpoint itself is capable of correct semantic reasoning on the same
  Thor hardware.
- TensorRT Edge-LLM Cosmos3 should be treated as **blocked for this project until a
  correctness root cause is identified or upstream behavior changes**.

The next runtime to evaluate is **FlashRT**, using its Thor-specific Cosmos3 reasoner
path as an independent implementation.

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
| Native build | Release, aarch64, `ENABLE_CUTE_DSL=ALL`, `TRT_PACKAGE_DIR=/usr` |

The project's native Edge-LLM plugin used during testing was:

```text
~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
```

### Cosmos3 checkpoint

Local checkpoint:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/hf_checkpoint
```

The copied checkpoint used for x86 export was transferred directly from this Thor
checkpoint so checkpoint revision/content was not intentionally changed between the
AArch64 and x86 export experiments.

### Known image fixture

```text
/tmp/vlm_multiframe_512/frame_001.jpg
SHA256: c221acdc6eef46309207dfa33c79708ca70b05b51e770375661308d3e6595acb
```

The image visibly contains a red panda on a wooden platform with green foliage.

---

## 3. Official TensorRT Edge-LLM reasoner workflow

For the base `nvidia/Cosmos3-Edge` reasoner, the pinned NVIDIA documentation uses the
standard Edge-LLM multimodal reasoning path:

```bash
tensorrt-edgellm-export \
  "$REASONING_CHECKPOINT" \
  "$ONNX_DIR/reasoning" \
  --task reasoning

./build/examples/llm/llm_build \
  --onnxDir "$ONNX_DIR/reasoning/llm" \
  --engineDir "$ENGINE_DIR/reasoning" \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096

./build/examples/multimodal/visual_build \
  --onnxDir "$ONNX_DIR/reasoning/visual" \
  --engineDir "$ENGINE_DIR/reasoning"

./build/examples/llm/llm_inference \
  --engineDir "$ENGINE_DIR/reasoning" \
  --multimodalEngineDir "$ENGINE_DIR/reasoning" \
  --inputFile input.json \
  --outputFile output.json
```

A later review of the pinned installation guide found an important platform split:

- **Export and quantization:** x86-64 Linux host.
- **C++ engine build and runtime:** target system such as Jetson Thor.
- Export itself is CPU-only; quantization is the step that requires a GPU.

This requirement was missed during the first Thor-native export and was later
explicitly retested using a clean x86-64 workflow (Section 8).

---

## 4. Raw Hugging Face reference control

Before blaming the checkpoint, image, or Thor platform, the original checkpoint was
run directly through Transformers on the same Thor and same image.

Environment used for the successful image reference:

```text
nvcr.io/nvidia/pytorch:26.05-py3
AutoModelForImageTextToText
AutoProcessor
model class: Cosmos3EdgeForConditionalGeneration
processor: Cosmos3EdgeProcessor
dtype: bfloat16
device_map: auto
do_sample: false
```

Image prompt:

```text
Describe the scene in the image.
```

The raw HF model correctly described a **red panda**, including the reddish-brown
fur, white facial markings, wooden platform, and green foliage.

This establishes that:

1. the checkpoint can interpret the fixture correctly;
2. the fixture itself is not the source of the `dog` result;
3. Thor is capable of running the model correctly outside Edge-LLM.

---

## 5. Standard ONNX -> TensorRT Edge-LLM result

A normal reasoning export and native Thor engine build completed successfully. The
runtime also completed inference without reporting an execution error.

However, for the red-panda fixture and prompt:

```text
Describe the scene in the image.
```

Edge-LLM incorrectly described a **dog**. Across tests the generated description
included details such as a black-and-white coat and a white surface / wooden deck.

A control using the same runtime input/image path with Cosmos-Reason2-2B correctly
identified the red panda. That reduced the likelihood of a generic image-file or
`llm_inference` message-format problem.

At this stage the failure initially looked visual, but later text-only testing showed
that conclusion was too narrow.

---

## 6. Direct-builder diagnostic path

The experimental checkpoint-direct builder was tested as an independent path around
ONNX export. This was useful diagnostically, but it exposed two separate artifact
problems before inference could run.

A disposable Python environment was used:

```text
/tmp/edgellm-direct-venv
```

Only environment-local Python dependencies were added; no JetPack/CUDA/TensorRT
system packages were changed.

### 6.1 Direct-builder issue: RoPE metadata schema mismatch

The direct builder emitted Cosmos3 text configuration using a Transformers-v5-style
nested shape under:

```text
text_config.rope_parameters
```

while the pinned C++ multimodal runtime expected normalized fields including:

```text
text_config.rope_theta
text_config.rope_scaling
```

The original direct runtime failed during initialization with a JSON type error
because `rope_theta` was missing at the expected location.

For diagnosis only, a copied engine directory was patched so that:

```text
rope_theta = 100000000
mrope_section = [24, 20, 20]
```

were visible in the form expected by the runtime. The original direct-builder
artifacts were left untouched.

### 6.2 Direct-builder issue: missing multimodal chat-template content types

After the RoPE workaround, multimodal input failed with:

```text
Unknown content type: image
EDGELLM_BAD_MEDIA_COUNT
```

The direct-builder artifact had an empty `content_types` map, whereas the working
ONNX-generated template defined image/video sentinels using the expected
`<|vision_start|>`, pad, and `<|vision_end|>` tokens.

For diagnosis, the known-good ONNX-generated `processed_chat_template.json` was
copied into the patched direct-engine directory.

### 6.3 Direct-builder semantic result

After both artifact workarounds, direct-builder multimodal inference ran to
completion but still described the red-panda image as a **dog**.

That result alone did not exonerate ONNX because the direct builder had already shown
independent integration problems. The later text-only control was more decisive.

---

## 7. Text-only control: the key diagnostic

To remove the entire visual stack from the experiment, the following exact text-only
prompt was used:

```text
What animal is known for reddish-brown fur, white facial markings, and a long
ringed tail? Reply with only the animal name.
```

### Raw HF result

The raw checkpoint reasoned through alternatives and semantically converged on
**red panda**. The response was verbose despite the requested format, but its semantic
conclusion was correct.

### Edge-LLM ONNX result

The standard Edge-LLM engine returned:

```text
Guanaco<|im_end|>
```

### Edge-LLM direct-builder result

The patched direct-builder engine returned the same result:

```text
Guanaco<|im_end|>
```

This changed the investigation materially. A text-only failure means the red-panda
image mismatch cannot be explained solely by image resize, patch ordering, visual
position embedding, PatchMerger behavior, or other visual preprocessing.

The common failure surface moved toward the Cosmos3 text-model conversion / model
representation / runtime contract.

---

## 8. Supported x86-64 export retest

Because NVIDIA explicitly documents export on x86-64, the reasoner was exported again
on an x86-64 Ubuntu host (`section9`) rather than Thor.

### 8.1 x86 host

```text
architecture: x86_64
RAM: 94 GiB
available RAM during setup: ~82 GiB
disk available: ~645 GiB
Docker: 29.2.1
```

A clean checkout was pinned to the same Edge-LLM revision:

```text
71dd1bae032e70771265917ec74d3ff4cad07a10
```

The exact Thor checkpoint was copied to the x86 host before export.

### 8.2 Export-container dependency lesson

The pinned Edge-LLM installation documentation recommends the NVIDIA PyTorch 25.12
container for the x86 export workflow.

The stock `25.12` container began with a matching NVIDIA Torch/torchvision pair, but
`pip install -e .` for the pinned Edge-LLM package upgraded Torch according to the
project's own dependency pins while leaving the preinstalled torchvision version
behind. This produced:

```text
RuntimeError: operator torchvision::nms does not exist
```

The pinned package metadata requires:

```text
torch==2.13.0
transformers==5.14.1
```

and its tools dependency set specifies:

```text
torchvision==0.28.0
```

After explicitly matching torchvision to `0.28.0`, the x86 export environment was:

```text
torch:        2.13.0+cu130
torchvision:  0.28.0+cu130
transformers: 5.14.1
cuda_available: False
```

The CPU-only export then completed successfully.

### 8.3 x86 export artifacts

The reasoner export produced approximately 4.6 GiB of ONNX artifacts including:

```text
reasoning/llm/model.onnx
reasoning/llm/model.onnx.data
reasoning/llm/embedding.safetensors
reasoning/llm/processed_chat_template.json
reasoning/visual/model.onnx
reasoning/visual/model.onnx.data
reasoning/visual/preprocessor_config.json
```

The exported tree was copied back to Thor into a separate user-owned directory:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx
```

A fresh LLM engine was built natively on Thor into:

```text
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning
```

### 8.4 x86-exported text result

The exact same text-only control still produced:

```text
Guanaco<|im_end|>
```

This is the strongest Edge-LLM result so far because it follows NVIDIA's documented
architecture split:

```text
x86-64 export -> transfer ONNX -> native Thor engine build -> Thor inference
```

The visual engine was intentionally not rebuilt from the x86 artifacts after this
text gate failed; a text-only failure already establishes a lower-level correctness
problem independent of vision.

---

## 9. Current correctness matrix

| Test | Raw HF | Edge-LLM ONNX (Thor export) | Edge-LLM ONNX (x86 export) | Edge-LLM direct |
|---|---|---|---|---|
| Text-only red-panda clue | red panda / semantically correct | `Guanaco` | `Guanaco` | `Guanaco` |
| Red-panda image | red panda | dog | not run; text gate failed | dog |
| Runtime completion | yes | yes | yes | yes after artifact workarounds |

The repeated `Guanaco` result across independently prepared Edge-LLM engines is more
important than the exact word itself: the failure is deterministic and shared by
multiple Edge-LLM preparation paths while the raw checkpoint remains semantically
correct.

---

## 10. What is ruled out vs. what remains open

### Strongly reduced / ruled out as primary cause

The available evidence argues against these as the primary explanation:

- corrupt or misleading red-panda fixture;
- generic `llm_inference` image-file handling;
- Thor being unable to execute the checkpoint correctly;
- AArch64 ONNX export being the sole cause;
- an ONNX-only conversion defect;
- a purely visual-preprocessing defect.

### Still plausible

The remaining high-value failure surface is in the Cosmos3 text model's translation
into the Edge-LLM representation/runtime, including possibilities such as:

- checkpoint weight/layer remapping;
- embedding or LM-head mapping;
- mRoPE configuration or application;
- Cosmos3/Nemotron-H decoder assumptions;
- squared-ReLU MLP implementation or graph lowering;
- tokenizer / special-token/runtime contract differences;
- another shared Cosmos3-specific Edge-LLM model integration issue.

No single one of these has yet been proven to be the root cause.

---

## 11. Cosmos3 text architecture facts relevant to a future RCA

The pinned Edge-LLM Cosmos3 reasoner text implementation describes the reasoner as a
standard dense mRoPE decoder with one important MLP departure from the shared default
CausalLM:

```text
down_proj(relu(up_proj(x)) ** 2)
```

rather than SwiGLU.

Relevant model constants documented/implemented by the Edge-LLM path include:

```text
28 combined decoder layers
hidden size: 2048
GQA heads: 16 query / 8 KV
head_dim: 128
rope_theta: 1e8
mrope_section: [24, 20, 20]
qk_norm_for_text: false
```

The native Nemotron-H checkpoint stores 56 alternating attention/MLP blocks that the
Edge-LLM implementation maps into a 28-layer attention+MLP view.

If the project later returns to deep Edge-LLM debugging, the preferred approach is
not more end-to-end prompt guessing. Instead, compare raw-HF and Edge-LLM-compatible
intermediate values from the earliest possible point:

```text
embeddings
  -> layer 0 attention
  -> layer 0 MLP
  -> subsequent layers
  -> final norm
  -> LM-head logits
```

The first divergence would localize the patch surface.

---

## 12. Why the project is pivoting to FlashRT

At this point Edge-LLM Cosmos3 has consumed enough investigation to establish a
reproducible correctness blocker, but not yet a root-cause patch. Continuing directly
into layer-level TensorRT RCA is possible, but it is no longer the highest-value next
experiment.

FlashRT provides an independent Cosmos3-Edge Reasoner implementation specifically for
Jetson AGX Thor / SM110. Its documented reasoner supports batch-1 greedy **text,
image, and video** inputs and uses:

- official Cosmos preprocessing/chat-template behavior;
- a parity-first Torch implementation for vision + prefill;
- Thor-specific optimized decode kernels;
- BF16 and FP4 decode paths.

For this project the immediate value is **independent correctness**, not performance.
The first FlashRT experiment should therefore use BF16 and the same controls already
used against Edge-LLM:

1. text-only red-panda clue;
2. known red-panda image fixture;
3. only after both pass, measure latency and evaluate image/video temporal inputs.

No ROS integration should occur until standalone correctness passes.

---

## 13. Runtime strategy going forward

There is no requirement that every project model use the same inference backend.
A likely architecture, if FlashRT validates, is:

```text
ROS2 VLM/reasoning abstraction
  |
  +-- Cosmos-Reason2 -> TensorRT Edge-LLM
  |
  +-- Cosmos3-Edge   -> FlashRT
```

The ROS-facing contract can remain model/backend agnostic: ordered observations and a
prompt enter the backend, and normalized text/structured reasoning exits it.

Cosmos-Reason2 Edge-LLM performance and correctness work should remain untouched by
the Cosmos3 runtime pivot.

---

## 14. Preserved artifacts on Thor

The following diagnostic artifacts were intentionally kept so the investigation can
be reproduced or resumed:

```text
Checkpoint:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/hf_checkpoint

Original NVIDIA-reference ONNX/TRT engine:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/nvidia-reference-reasoning

Direct-builder original:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/direct-reference

Direct-builder diagnostic copy with metadata/template workarounds:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/direct-reference-ropefix

Supported x86-exported ONNX tree:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/x86-reference-reasoning-onnx

Fresh Thor LLM engine from x86 export:
~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines/x86-reference-reasoning

Text-only input:
/tmp/cosmos3_text_only.json

Edge-LLM x86-exported text result:
/tmp/cosmos3_x86_text_only_output.json

Known image:
/tmp/vlm_multiframe_512/frame_001.jpg
```

Do not treat `/tmp` paths as durable project storage; copy any fixture/result needed
for long-term reproduction into the repository or benchmark artifact store.

---

## 15. Decision record

As of 2026-08-24:

- **Cosmos3-Edge checkpoint:** viable; raw HF reference behaves correctly.
- **TensorRT Edge-LLM Cosmos3 reasoner:** blocked for project use by reproducible
  semantic correctness failure.
- **Original Thor/AArch64 ONNX export:** unsupported by NVIDIA's documented export
  workflow; retained only as historical evidence.
- **Supported x86-64 export:** completed successfully and reproduces the same text
  failure after native Thor engine build.
- **Experimental direct builder:** useful diagnostic, but required two artifact
  workarounds and still reproduced the semantic failure.
- **Next action:** evaluate FlashRT Cosmos3 Reasoner on Thor, BF16 correctness first.
- **Deferred action:** layer-by-layer Edge-LLM RCA / upstream bug report if FlashRT
  validates and there is still value in repairing the Edge-LLM backend.

This status should remain explicit in model-selection and benchmarking code: do not
include TensorRT Edge-LLM Cosmos3 results in temporal-performance comparisons as if
they represented a validated model until correctness is restored.
