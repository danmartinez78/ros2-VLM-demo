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
  (`standard_vlm`), the same `llm_inference` / `edge_vlm_server` path used
  for Cosmos-Reason2.  This is **not** the Cosmos3 policy runtime
  (`cosmos3_policy_inference`).
- **Checkpoint family:** `cosmos3-edge` (distinct from Cosmos-Reason2 and from
  Cosmos3-Edge-Policy).

The distinction matters for provenance: both Cosmos-Reason2 and Cosmos3-Edge
use the standard Edge-LLM VLM runtime.  Cosmos3-Edge preparation and build
use the `cosmos3_edge` strategy; the runtime strategy for both models is
`standard_vlm` (see `scripts/models/engine_profiles.json`).

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

1. Export checkpoint to ONNX via the pinned `tensorrt-edgellm-export` tool:
   `tensorrt-edgellm-export <checkpoint> <onnx_dir/reasoning> --task reasoning`
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

The pinned TRT Edge-LLM 0.10.0 guide defines the Cosmos3-Edge reasoning path
as **image + prompt → text**, using the standard image message format.  The
exact multi-frame / native-video input contract for the reasoner is **not
confirmed** at this revision; the `--video frame_...` examples in the upstream
guide apply to the policy runtime, not the reasoning path.

Phase 2 smoke tests therefore start with the documented single-image (F1) path:

```
llm_inference --engineDir <llm_dir> --multimodalEngineDir <visual_dir> \
              --inputFile <smoke_input_f1.json> --outputFile <smoke_output_f1.json>
```

where `smoke_input_f1.json` uses the documented image message format.

Multi-image or native-video smoke tests will be added in a subsequent phase
after the Cosmos3 reasoner parser/runner contract is confirmed on hardware.

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
| `runtime_strategy` | `standard_vlm` (base) or `cosmos3_policy_inference` (policy) |
| `cosmos3_native_stages` | Array of native stage names reported by the runtime |
| `temporal_input.frame_count` | Number of frames submitted |
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
2. ONNX export for the reasoning path (`tensorrt-edgellm-export … --task reasoning`).
3a. LLM engine build (`llm_build`).
3b. Visual engine build (`visual_build`).
4. Smoke inference, single frame / image (F1) (`llm_inference --engineDir … --inputFile …`).
5. Provenance / manifest capture (`modelctl`).

No large downloads or builds are performed on CI.  All commands are printed
to stdout and validated for structural correctness only.

---

*Document generated as part of Phase 1 scaffold (issue #77).*  
*Update after Phase 2 hardware execution with measured latency, stage names,
and build artifacts.*
