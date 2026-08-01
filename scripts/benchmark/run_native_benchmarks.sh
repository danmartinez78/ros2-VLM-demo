#!/usr/bin/env bash
# run_native_benchmarks.sh
#
# Thin wrapper around NVIDIA TensorRT Edge-LLM native benchmarking tools.
#
# Delegates entirely to NVIDIA's tools:
#   - llm_bench --mode prefill   (prefill latency / throughput)
#   - llm_bench --mode decode    (generation throughput)
#   - llm_bench --mode visual    (vision-encoder latency / throughput)
#   - llm_inference --dumpProfile --profileOutputFile ...
#
# This script does NOT reimplement TTFT, token throughput, ViT timing, or any
# layer profiling — those are authoritative in the NVIDIA tool outputs.
# Native JSON/profile artifacts are preserved verbatim.
#
# Reference: https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html
#
# Usage
# -----
#   source scripts/cosmos_env.sh          # sets COSMOS_* and EDGELLM_* env vars
#   bash scripts/benchmark/run_native_benchmarks.sh [OPTIONS]
#
# Options
#   --output-dir DIR        Directory to write artifacts (default: /tmp/cosmos_native_bench_TIMESTAMP)
#   --warmup N              Number of warm-up iterations (default: 3)
#   --iterations N          Number of measured iterations (default: 10)
#   --max-generate-length N Token budget for decode/end-to-end (default: 64)
#   --input-image PATH      Path to a representative input image (default: env IMAGE_PATH or skip)
#   --input-vlm-json PATH   Path to llm_inference input JSON (default: env INPUT_VLM_JSON or skip)
#   --skip-prefill          Skip llm_bench --mode prefill
#   --skip-decode           Skip llm_bench --mode decode
#   --skip-visual           Skip llm_bench --mode visual
#   --skip-profile          Skip llm_inference --dumpProfile
#   --dry-run               Print commands without executing them

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/tmp/cosmos_native_bench_${TIMESTAMP}"
WARMUP=3
ITERATIONS=10
MAX_GENERATE_LENGTH=64
INPUT_IMAGE="${IMAGE_PATH:-}"
INPUT_VLM_JSON="${INPUT_VLM_JSON:-}"
SKIP_PREFILL=false
SKIP_DECODE=false
SKIP_VISUAL=false
SKIP_PROFILE=false
DRY_RUN=false

# ── argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)       OUTPUT_DIR="$2";           shift 2 ;;
    --warmup)           WARMUP="$2";               shift 2 ;;
    --iterations)       ITERATIONS="$2";           shift 2 ;;
    --max-generate-length) MAX_GENERATE_LENGTH="$2"; shift 2 ;;
    --input-image)      INPUT_IMAGE="$2";          shift 2 ;;
    --input-vlm-json)   INPUT_VLM_JSON="$2";       shift 2 ;;
    --skip-prefill)     SKIP_PREFILL=true;         shift   ;;
    --skip-decode)      SKIP_DECODE=true;          shift   ;;
    --skip-visual)      SKIP_VISUAL=true;          shift   ;;
    --skip-profile)     SKIP_PROFILE=true;         shift   ;;
    --dry-run)          DRY_RUN=true;              shift   ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── environment validation ────────────────────────────────────────────────────

: "${TENSORRT_EDGE_LLM_ROOT:?Set TENSORRT_EDGE_LLM_ROOT (source scripts/cosmos_env.sh)}"
: "${COSMOS_LLM_ENGINE_DIR:?Set COSMOS_LLM_ENGINE_DIR (source scripts/cosmos_env.sh)}"
: "${COSMOS_MULTIMODAL_ENGINE_DIR:?Set COSMOS_MULTIMODAL_ENGINE_DIR (source scripts/cosmos_env.sh)}"
: "${EDGELLM_PLUGIN_PATH:?Set EDGELLM_PLUGIN_PATH (source scripts/cosmos_env.sh)}"

LLM_BENCH="${TENSORRT_EDGE_LLM_ROOT}/build/examples/llm/llm_bench"
LLM_INFERENCE="${TENSORRT_EDGE_LLM_ROOT}/build/examples/llm/llm_inference"

if [[ ! -x "${LLM_BENCH}" ]]; then
  echo "ERROR: llm_bench not found or not executable: ${LLM_BENCH}" >&2
  echo "       Build TensorRT Edge-LLM first." >&2
  exit 1
fi
if [[ ! -x "${LLM_INFERENCE}" ]]; then
  echo "ERROR: llm_inference not found or not executable: ${LLM_INFERENCE}" >&2
  exit 1
fi

# ── output directory ──────────────────────────────────────────────────────────

if [[ "${DRY_RUN}" == "false" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  echo "Benchmark artifacts: ${OUTPUT_DIR}"
else
  echo "[DRY RUN] Would write to: ${OUTPUT_DIR}"
fi

# ── helpers ───────────────────────────────────────────────────────────────────

ERRORS=()
SKIPPED_MODES=()

run_cmd() {
  local label="$1"; shift
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] ${label}: $*"
    return 0
  fi
  echo "==> ${label}"
  echo "    $*"
  if ! "$@"; then
    echo "WARNING: ${label} exited with non-zero status" >&2
    ERRORS+=("${label} failed")
    return 1
  fi
  return 0
}

# ── metadata collection ───────────────────────────────────────────────────────

collect_metadata() {
  local arch kernel jetpack cuda trt gpu_cc gpu_name edge_llm_commit edge_llm_tag nvpmodel
  arch=$(uname -m 2>/dev/null || echo "unknown")
  kernel=$(uname -r 2>/dev/null || echo "unknown")
  jetpack=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || echo "")
  cuda=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "")
  trt=$(dpkg-query -W -f='${Version}' libnvinfer-dev 2>/dev/null || echo "")
  gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
  edge_llm_commit=$(git -C "${TENSORRT_EDGE_LLM_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "")
  edge_llm_tag=$(git -C "${TENSORRT_EDGE_LLM_ROOT}" describe --tags --abbrev=0 2>/dev/null || echo "")
  nvpmodel=$(nvpmodel -q 2>/dev/null | head -1 || echo "")

  python3 -c "
import json, sys
print(json.dumps({
  'arch': '${arch}',
  'kernel': '${kernel}',
  'jetpack_version': '${jetpack}' or None,
  'cuda_version': '${cuda}' or None,
  'tensorrt_version': '${trt}' or None,
  'gpu_compute_capability': '${gpu_cc}' or None,
  'gpu_name': '${gpu_name}' or None,
  'edge_llm_commit': '${edge_llm_commit}' or None,
  'edge_llm_version_tag': '${edge_llm_tag}' or None,
  'nvpmodel_mode': '${nvpmodel}' or None,
  'model_name': '${COSMOS_MODEL_NAME:-}' or None,
  'llm_engine_dir': '${COSMOS_LLM_ENGINE_DIR}',
  'multimodal_engine_dir': '${COSMOS_MULTIMODAL_ENGINE_DIR}',
  'max_generate_length': ${MAX_GENERATE_LENGTH},
  'warmup_iterations': ${WARMUP},
  'measured_iterations': ${ITERATIONS},
  'input_image_path': '${INPUT_IMAGE}' or None,
  'input_vlm_json': '${INPUT_VLM_JSON}' or None,
}, indent=2, sort_keys=True))
"
}

# ── run llm_bench for each mode ───────────────────────────────────────────────

run_llm_bench_mode() {
  local mode="$1"
  local out_file="${OUTPUT_DIR}/llm_bench_${mode}.txt"

  local cmd=(
    "${LLM_BENCH}"
    --mode "${mode}"
    --engineDir "${COSMOS_LLM_ENGINE_DIR}"
    --warmUp "${WARMUP}"
    --numRuns "${ITERATIONS}"
  )
  if [[ "${mode}" == "visual" ]]; then
    cmd+=(--multimodalEngineDir "${COSMOS_MULTIMODAL_ENGINE_DIR}")
    if [[ -n "${INPUT_IMAGE}" ]]; then
      cmd+=(--inputImage "${INPUT_IMAGE}")
    fi
  fi

  if run_cmd "llm_bench --mode ${mode}" "${cmd[@]}" 2>&1 | tee "${out_file}"; then
    echo "${mode}_bench_file=${out_file}"
    return 0
  fi
  return 1
}

PREFILL_OUT=""
DECODE_OUT=""
VISUAL_OUT=""
PROFILE_OUT=""
INFERENCE_OUT=""

# ── prefill ───────────────────────────────────────────────────────────────────

if [[ "${SKIP_PREFILL}" == "true" ]]; then
  SKIPPED_MODES+=("prefill")
else
  if [[ "${DRY_RUN}" == "false" ]]; then
    echo "--- prefill benchmark ---"
    PREFILL_OUT="${OUTPUT_DIR}/llm_bench_prefill.txt"
    run_cmd "llm_bench --mode prefill" \
      "${LLM_BENCH}" \
      --mode prefill \
      --engineDir "${COSMOS_LLM_ENGINE_DIR}" \
      --warmUp "${WARMUP}" \
      --numRuns "${ITERATIONS}" \
      2>&1 | tee "${PREFILL_OUT}" || true
  else
    echo "[DRY RUN] llm_bench --mode prefill --engineDir ${COSMOS_LLM_ENGINE_DIR} --warmUp ${WARMUP} --numRuns ${ITERATIONS}"
    PREFILL_OUT="llm_bench_prefill.txt"
  fi
fi

# ── decode ────────────────────────────────────────────────────────────────────

if [[ "${SKIP_DECODE}" == "true" ]]; then
  SKIPPED_MODES+=("decode")
else
  if [[ "${DRY_RUN}" == "false" ]]; then
    echo "--- decode benchmark ---"
    DECODE_OUT="${OUTPUT_DIR}/llm_bench_decode.txt"
    run_cmd "llm_bench --mode decode" \
      "${LLM_BENCH}" \
      --mode decode \
      --engineDir "${COSMOS_LLM_ENGINE_DIR}" \
      --warmUp "${WARMUP}" \
      --numRuns "${ITERATIONS}" \
      --maxGenerateLength "${MAX_GENERATE_LENGTH}" \
      2>&1 | tee "${DECODE_OUT}" || true
  else
    echo "[DRY RUN] llm_bench --mode decode --engineDir ${COSMOS_LLM_ENGINE_DIR} --warmUp ${WARMUP} --numRuns ${ITERATIONS} --maxGenerateLength ${MAX_GENERATE_LENGTH}"
    DECODE_OUT="llm_bench_decode.txt"
  fi
fi

# ── visual ────────────────────────────────────────────────────────────────────

if [[ "${SKIP_VISUAL}" == "true" ]]; then
  SKIPPED_MODES+=("visual")
else
  if [[ "${DRY_RUN}" == "false" ]]; then
    echo "--- visual encoder benchmark ---"
    VISUAL_OUT="${OUTPUT_DIR}/llm_bench_visual.txt"
    VISUAL_CMD=(
      "${LLM_BENCH}"
      --mode visual
      --engineDir "${COSMOS_LLM_ENGINE_DIR}"
      --multimodalEngineDir "${COSMOS_MULTIMODAL_ENGINE_DIR}"
      --warmUp "${WARMUP}"
      --numRuns "${ITERATIONS}"
    )
    if [[ -n "${INPUT_IMAGE}" ]]; then
      VISUAL_CMD+=(--inputImage "${INPUT_IMAGE}")
    fi
    run_cmd "llm_bench --mode visual" "${VISUAL_CMD[@]}" 2>&1 | tee "${VISUAL_OUT}" || true
  else
    echo "[DRY RUN] llm_bench --mode visual --engineDir ${COSMOS_LLM_ENGINE_DIR} --multimodalEngineDir ${COSMOS_MULTIMODAL_ENGINE_DIR} ..."
    VISUAL_OUT="llm_bench_visual.txt"
  fi
fi

# ── end-to-end profiling with llm_inference --dumpProfile ────────────────────

if [[ "${SKIP_PROFILE}" == "true" ]]; then
  SKIPPED_MODES+=("profile")
else
  if [[ -z "${INPUT_VLM_JSON}" ]]; then
    echo "WARNING: --input-vlm-json not set; skipping llm_inference --dumpProfile" >&2
    SKIPPED_MODES+=("profile")
  else
    PROFILE_OUT="${OUTPUT_DIR}/llm_inference_profile.json"
    INFERENCE_OUT="${OUTPUT_DIR}/llm_inference_output.json"
    if [[ "${DRY_RUN}" == "false" ]]; then
      echo "--- end-to-end profiling ---"
      run_cmd "llm_inference --dumpProfile" \
        "${LLM_INFERENCE}" \
        --engineDir "${COSMOS_LLM_ENGINE_DIR}" \
        --multimodalEngineDir "${COSMOS_MULTIMODAL_ENGINE_DIR}" \
        --inputFile "${INPUT_VLM_JSON}" \
        --outputFile "${INFERENCE_OUT}" \
        --maxGenerateLength "${MAX_GENERATE_LENGTH}" \
        --dumpProfile \
        --profileOutputFile "${PROFILE_OUT}" || true
    else
      echo "[DRY RUN] llm_inference --engineDir ... --dumpProfile --profileOutputFile ${PROFILE_OUT}"
    fi
  fi
fi

# ── write manifest JSON ───────────────────────────────────────────────────────

RUN_ID="native_bench_${TIMESTAMP}"
RECORDED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

manifest_file="${OUTPUT_DIR}/manifest.json"

if [[ "${DRY_RUN}" == "false" ]]; then
  METADATA_JSON=$(collect_metadata 2>/dev/null || echo '{}')

  python3 - "${manifest_file}" <<PYEOF
import json, sys

manifest = {
  "schema_version": "1",
  "run_id": "${RUN_ID}",
  "recorded_at": "${RECORDED_AT}",
  "metadata": ${METADATA_JSON},
  "llm_bench_prefill": "$(basename "${PREFILL_OUT}")" if "${PREFILL_OUT}" else None,
  "llm_bench_decode": "$(basename "${DECODE_OUT}")" if "${DECODE_OUT}" else None,
  "llm_bench_visual": "$(basename "${VISUAL_OUT}")" if "${VISUAL_OUT}" else None,
  "llm_inference_profile": "$(basename "${PROFILE_OUT}")" if "${PROFILE_OUT}" else None,
  "llm_inference_output": "$(basename "${INFERENCE_OUT}")" if "${INFERENCE_OUT}" else None,
  "skipped_modes": ${SKIPPED_MODES@Q},
  "errors": [],
}

with open(sys.argv[1], "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
print("Manifest written:", sys.argv[1])
PYEOF

  echo ""
  echo "Native benchmark run complete."
  echo "  Run ID:   ${RUN_ID}"
  echo "  Output:   ${OUTPUT_DIR}"
  echo "  Manifest: ${manifest_file}"
  if [[ "${#SKIPPED_MODES[@]}" -gt 0 ]]; then
    echo "  Skipped: ${SKIPPED_MODES[*]}"
  fi
  if [[ "${#ERRORS[@]}" -gt 0 ]]; then
    echo "  Errors:  ${ERRORS[*]}"
  fi
else
  echo "[DRY RUN] Would write manifest: ${manifest_file}"
fi
