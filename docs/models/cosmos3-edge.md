# Cosmos3-Edge: TensorRT Edge-LLM 0.10.0 Runtime Audit

**Status:** Phase 1 scaffold — hardware-independent audit.  
No model has been downloaded. No TensorRT engine has been built.  
All conclusions are derived from TensorRT Edge-LLM 0.10.0 documentation,
checkpoint configuration metadata, and upstream source inspection.

---

## 1. Upstream revision inspected

| Item | Value |
|---|---|
| Repository | `https://github.com/NVIDIA/TensorRT-Edge-LLM.git` |
| Pinned commit (project manifest) | `71dd1bae032e70771265917ec74d3ff4cad07a10` |
| Version tag | `0.10.0` |
| Manifest file | `scripts/thor/jp72_manifest.json` (`edge_llm.commit`) |

All statements below refer specifically to this revision unless noted.

---

## 2. Supported checkpoints

TensorRT Edge-LLM 0.10.0 lists the following Cosmos3 checkpoints in its
supported-model matrix:

| HuggingFace ID | Use case |
|---|---|
| `nvidia/Cosmos3-Edge` | Scene understanding / multimodal reasoning (VLM) |
| `nvidia/Cosmos3-Edge-Policy-DROID` | Robot action policy (policy runtime) |

This document covers **`nvidia/Cosmos3-Edge`** only.  
`nvidia/Cosmos3-Edge-Policy-DROID` uses a distinct policy runtime with
component layout `und_prefill`, `gen`, `vae_encoder` and is **not** addressed
here.

---

## 3. Model architecture

`nvidia/Cosmos3-Edge` is a video-native multimodal VLM.  Key architecture
properties:

- **Modality:** native video (natively video-aware; not a list of independent
  still images).
- **Runtime class:** standard Edge-LLM VLM multimodal runtime
  (`cosmos3_edge_vlm`), **not** the Cosmos3 policy runtime
  (`cosmos3_policy_inference`).
- **Checkpoint family:** `cosmos3-edge` (distinct from Cosmos-Reason2 and from
  Cosmos3-Edge-Policy).

The distinction matters for provenance: Cosmos-Reason2 encodes frames as
independent images; Cosmos3-Edge encodes them as a single stacked video
`ImageData` object with temporal metadata (see
`src/tensorrt_edge_llm_backend.cpp`).

---

## 4. Engine components and expected paths

For the base `nvidia/Cosmos3-Edge` checkpoint the supported component layout
is identical to Cosmos-Reason2 (two-component VLM):

| Component | Kind | Relative engine directory |
|---|---|---|
| `llm` | `llm` | `llm/` |
| `visual` | `visual` | `visual/` |

Runtime engine root (managed profile `cosmos3-edge-thor-f8`):

```
${EDGE_VLM_WORKSPACE_DIR}/Cosmos3-Edge/engines/cosmos3-edge-thor-f8/
  llm/       ← LLM TRT engine
  visual/    ← visual/multimodal TRT engine
```

These paths are **documented targets only**. No engine has been built; no
directory should be created until Phase 2 on Jetson AGX Thor.

---

## 5. Build frontend

| Path | Status |
|---|---|
| ONNX export + component-specific C++ builders (`llm_build`, `visual_build`) | **Supported** |
| Experimental direct builder for Cosmos3 policy | Experimental, policy checkpoint only |

For `nvidia/Cosmos3-Edge` (base VLM checkpoint) the supported workflow is:

1. Quantize/export checkpoint to ONNX via ModelOpt in the NVIDIA PyTorch
   container (`nvcr.io/nvidia/pytorch:26.05-py3`, ModelOpt 0.45.0).
2. Build LLM engine: `llm_build --onnxDir … --engineDir …`
3. Build visual engine: `visual_build --onnxDir … --engineDir …`

The experimental direct builder path documented for Cosmos3 policy
(`und_prefill`, `gen`, `vae_encoder`) does **not** apply to this checkpoint.

---

## 6. Precision / quantization

| Property | Value |
|---|---|
| Required precision | `nvfp4` |
| Checkpoint format | HuggingFace safetensors |
| Quantization toolchain | NVIDIA ModelOpt 0.45.0 |
| PyTorch container | `nvcr.io/nvidia/pytorch:26.05-py3` |

Same precision as Cosmos-Reason2-8B/2B.  This is a constraint imposed by the
TRT-Edge-LLM 0.10.0 support matrix for this model family on Jetson.

---

## 7. Required C++ runtime executable

The primary inference runtime for `nvidia/Cosmos3-Edge` is the same
Edge-LLM multimodal runtime used for Cosmos-Reason2:

```
llm_inference  (or the equivalent IPC/socket server: edge_vlm_server)
```

The policy-specific `cosmos3_policy_inference` executable is **not** needed
for the base checkpoint.

---

## 8. Image / video input and temporal metadata

Cosmos3-Edge processes input as a **native video sequence**, not as an ordered
list of independent images.  The upstream contract (already implemented in this
project for temporal VLM requests) is:

- Frames packed into a single stacked `ImageData` object.
- A `video` content item carrying frame count, FPS, and per-frame timestamps.
- Temporal metadata fields: `frame_count`, `fps`, `timestamps_ms`.

This is identical to the temporal encoding already used for Cosmos-Reason2 in
`src/tensorrt_edge_llm_backend.cpp` (lines 336–349) and is preserved without
change for Cosmos3-Edge.

For Phase 2 smoke tests, the following frame counts must be verified against
hardware:

- F4 (4 frames, temporal window)
- F8 (8 frames, temporal window)

---

## 9. Output / task mode

`nvidia/Cosmos3-Edge` produces **text reasoning output** (scene understanding,
spatial/temporal reasoning).  It is **not** action/policy-oriented; it does not
emit robot joint-space actions or continuous motor commands.

The Edge-LLM runtime path for this checkpoint exposes text generation output
directly via the standard multimodal VLM inference API.

---

## 10. Thor / JetPack / TensorRT constraints

| Constraint | Value / Status |
|---|---|
| Target hardware | Jetson AGX Thor (aarch64) |
| JetPack / L4T | L4T R39.2.x (JetPack 7.2) |
| CUDA toolkit | 13.0 |
| CUDA compute family | 13.2 (Thor GPU) |
| TensorRT (bundled with Edge-LLM 0.10.0) | Per Edge-LLM build requirements |

These constraints are inherited from the project's pinned JP 7.2 manifest.
They apply equally to Cosmos3-Edge and Cosmos-Reason2.

---

## 11. Native timing stages

Unlike Cosmos-Reason2 where benchmark tooling labels stages `vision`, `prefill`,
and `generation`, the stages visible in Cosmos3-Edge inference may differ.
Until hardware execution is completed these stage names must remain
**unresolved**.

The benchmark/provenance schema supports the following Cosmos3-Edge-specific
fields to avoid mislabeling:

| Schema field | Description |
|---|---|
| `runtime_strategy` | `cosmos3_edge_vlm` (base) or `cosmos3_policy_inference` (policy) |
| `cosmos3_native_stages` | Array of native stage names reported by the runtime |
| `temporal_input.frame_count` | Number of video frames submitted |
| `temporal_input.fps` | Frame rate of submitted video |
| `temporal_input.timestamps_ms` | Per-frame timestamps in milliseconds |
| `task_mode` | `text_reasoning` or `action_policy` |

These fields are additive; they do not replace or remove existing CR2 fields.

---

## 12. What remains unknown until hardware execution

The following cannot be determined from documentation or source inspection
alone:

- Actual engine file sizes and build time on Thor.
- Actual latency for F4/F8 video inputs.
- Exact stage names emitted by `cosmos3_edge_vlm` runtime profiling.
- Whether NVFP4 quantization is lossless for scene-understanding tasks at our
  target resolution and frame counts.
- Maximum supported frame count before context-length overflow.
- Whether `llm_bench` mode names (`prefill`, `decode`, `visual`) map 1:1 to
  Cosmos3-Edge internal stages or require a remapping layer.

---

## 13. Relationship to Cosmos3-Edge-Policy-DROID (non-goal)

This document covers **only** the base `nvidia/Cosmos3-Edge` checkpoint for
scene understanding.  The following are explicitly out of scope for Phase 1:

- `nvidia/Cosmos3-Edge-Policy-DROID` policy engine build.
- `cosmos3_policy_inference` runtime components (`und_prefill`, `gen`,
  `vae_encoder`).
- Robot action / joint-space output integration.
- Reuse of DROID throughput figures for text-reasoning latency estimation.

---

## 14. Phase 2 procedure (generated dry-run)

The dry-run commands for Phase 2 on Jetson AGX Thor are emitted by:

```bash
python3 scripts/models/cosmos3_edge_commands.py --dry-run
```

This generates, in order:

1. Checkpoint acquisition (`huggingface-cli download`).
2. ONNX export / quantization (ModelOpt container).
3. LLM engine build (`llm_build`).
4. Visual engine build (`visual_build`).
4. Smoke inference, single frame (F1) (`llm_inference --frameCount 1`).
4. Smoke inference, F4 native-video (`llm_inference --frameCount 4`).
4. Smoke inference, F8 native-video (`llm_inference --frameCount 8`).
5. Provenance / manifest capture (`modelctl`).

No large downloads or builds are performed on CI.  All commands are printed
to stdout and validated for structural correctness only.

---

*Document generated as part of Phase 1 scaffold (issue #77).*  
*Update after Phase 2 hardware execution with measured latency, stage names,
and build artifacts.*
