# Cosmos3-Edge on Jetson Thor: validation findings

**Status:** TensorRT Edge-LLM text and patched single-image paths validated; FlashRT BF16 single-image and native-video paths validated.  
**Last updated:** 2026-08-25  
**Target:** `nvidia/Cosmos3-Edge` on NVIDIA Jetson AGX Thor.

This document records the runtime findings that matter for the generic ROS 2 VLM pipeline.

## Executive summary

Two early Cosmos3 Edge-LLM failures were isolated to representation issues rather than a fundamentally broken model runtime.

### Text prompt issue

Early text-only requests repeatedly returned `Guanaco`. Cross-runtime controls showed that the malformed prompt ended with a pre-closed `<think></think>` generation suffix. Feeding the exact same token sequence to raw Hugging Face, Edge-LLM, and FlashRT reproduced the same answer.

Using the official thinking-mode suffix ending in `<think>\n` restored coherent text reasoning. The Edge-LLM text decoder is therefore considered operational when the prompt template is applied correctly.

### Visual patch-layout issue

The original Edge-LLM visual path produced a corrupted interpretation of a known red-panda fixture. The root cause was a within-patch layout mismatch:

```text
Cosmos3 HF processor / trained patch embedding:
    16 x 16 patch flattened as H, W, C

Inherited Edge-LLM Qwen-style preprocessing:
    16 x 16 patch flattened as C, H, W
```

Direct reconstruction showed a substantial tensor mismatch. Reordering only the within-patch CHW representation to HWC made the reconstructed tensors bit-identical to the Hugging Face processor output.

The exported ONNX patch-embedding weight remained effectively identical to the checkpoint, proving the export did not compensate for the runtime packing difference.

A diagnostic ONNX with the patch-embedding columns permuted so that

```text
W_patched @ x_CHW == W_original @ x_HWC
```

restored correct single-image understanding in TensorRT Edge-LLM.

## Validated environment

| Item | Value |
| --- | --- |
| Hardware | NVIDIA Jetson AGX Thor |
| JetPack / L4T | JetPack 7.2 / L4T R39.2.x |
| TensorRT Edge-LLM | 0.10-era Thor build with SM110 support |
| FlashRT | Thor Cosmos3 path |
| Checkpoint | Cosmos3 Edge reasoner / understanding checkpoint |

## Reference image control

The known fixture is a 512 x 384 RGB image containing a red panda on a wooden platform with green foliage.

For Cosmos3 with `patch_size=16` and `merge_size=2`:

```text
patch grid:          32 x 24
raw patches/image:   768
merged tokens/image: 192
```

Raw Hugging Face correctly identifies the red panda and establishes the checkpoint/fixture/hardware correctness reference.

## FlashRT controls

FlashRT BF16 passed the same red-panda single-image correctness gate and became an independent runtime control.

The later native-video work also validated FlashRT/Cosmos3 temporal inference with exact timestamp metadata, controlled forward/reverse ordering, shuffled-order perturbation, and repeated-static controls. See [`../temporal-chronology-results.md`](../temporal-chronology-results.md).

## TensorRT Edge-LLM single-image result

After the patch-layout compensation, TensorRT Edge-LLM correctly described:

- the red panda fixture;
- its reddish-brown/white visual characteristics;
- the wooden platform and foliage;
- NVIDIA's robotic-arm reasoning sample.

This is the end-to-end evidence that the original single-image visual failure was caused by the patch-layout contract mismatch.

## Multi-image status

Independent multi-image requests were validated with 1, 2, and 3 images. A separate image-to-position binding issue was observed at 4 independent images. Engine token capacity was not the limiting factor.

That issue is separate from native-video inference and should not be conflated with the temporal path.

## Runtime selection implications

For the ROS-facing pipeline:

1. Preserve the runtime/process abstraction rather than coupling ROS directly to one inference library.
2. Keep raw Hugging Face controls available when validating a new runtime representation.
3. Treat prompt templates and visual preprocessing layout as first-class correctness contracts.
4. Validate native temporal correctness before drawing latency or semantic-quality conclusions.
5. Use controlled chronological, reversed, shuffled, repeated-static, and single-frame diagnostics for video-capable backends.
6. Compare TensorRT Edge-LLM, FlashRT, and other supported runtimes behind the same ROS/IPC abstraction.

## Current conclusion

Cosmos3 is usable as a generic VLM backend on Thor when the selected runtime applies the expected prompt and visual representation contracts. FlashRT currently provides the validated native-video path used by the temporal experiments; TensorRT Edge-LLM remains a viable single-image/runtime candidate after correcting the patch-layout mismatch.
