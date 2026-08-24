#!/usr/bin/env bash
# run_native_benchmarks.sh
#
# Thin wrapper around NVIDIA TensorRT Edge-LLM native benchmarking tools.
#
# Delegates entirely to NVIDIA's tools:
#   - llm_bench --mode prefill   (prefill latency / throughput)
#   - llm_bench --mode decode    (generation throughput)
#   - llm_bench --mode visual    (vision-encoder latency / throughput)
#   - llm_inference --warmup ... --dumpProfile --profileOutputFile ...
#
# This script does NOT reimplement TTFT, token throughput, ViT timing, or any
# layer profiling — those are authoritative in the NVIDIA tool outputs.
# Native JSON/profile artifacts are preserved verbatim.
#
# Reference: https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html
#
# Usage
# -----
#   source scripts/edge_vlm_env.sh          # sets EDGE_VLM_* and EDGELLM_* env vars
#   bash scripts/benchmark/run_native_benchmarks.sh [OPTIONS]
#
# Options
#   --output-dir DIR        Directory to write artifacts (default: /tmp/cosmos_native_bench_TIMESTAMP)
#   --batch-size N          Batch size for prefill and decode (default: 1)
#   --input-len N           Input token length for prefill benchmark (default: 2048)
#   --past-kv-len N         Past KV-cache length for decode benchmark (default: 2048)
#   --image-size HxW        Image dimensions for visual encoder benchmark (default: 1024x2048)
#   --warmup N              Warmup iterations for llm_bench modes (default: 3)
#   --iterations N          Measured iterations for llm_bench modes (default: 10)
#   --inference-warmup N    Warmup runs for llm_inference (default: 10)
#   --max-generate-length N Token budget for decode/end-to-end (default: 64)
#   --input-vlm-json PATH   Path to llm_inference input JSON (default: env INPUT_VLM_JSON or skip)
#   --skip-prefill          Skip llm_bench --mode prefill
#   --skip-decode           Skip llm_bench --mode decode
#   --skip-visual           Skip llm_bench --mode visual
#   --skip-profile          Skip llm_inference --dumpProfile
#   --quick                 Use faster smoke-test parameters (not the NVIDIA published workload)
#   --dry-run               Print commands without executing them

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ─────────────────────────────────────────────────────────────────

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/tmp/cosmos_native_bench_${TIMESTAMP}"
BATCH_SIZE=1
INPUT_LEN=2048
PAST_KV_LEN=2048
IMAGE_SIZE="1024x2048"
WARMUP=3
ITERATIONS=10
INFERENCE_WARMUP=10
MAX_GENERATE_LENGTH=64
INPUT_VLM_JSON="${INPUT_VLM_JSON:-}"
SKIP_PREFILL=false
SKIP_DECODE=false
SKIP_VISUAL=false
SKIP_PROFILE=false
DRY_RUN=false
RESOLVED_MODEL_NAME="${EDGE_VLM_MODEL_NAME:-}"
RESOLVED_ENGINE_PROFILE_ID="${EDGE_VLM_ENGINE_PROFILE_ID:-}"
RESOLVED_LLM_ENGINE_DIR="${EDGE_VLM_LLM_ENGINE_DIR:-}"
RESOLVED_MULTIMODAL_ENGINE_DIR="${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}"
ENGINE_PROVENANCE_JSON="{}"

# ── argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)          OUTPUT_DIR="$2";            shift 2 ;;
    --batch-size)          BATCH_SIZE="$2";             shift 2 ;;
    --input-len)           INPUT_LEN="$2";              shift 2 ;;
    --past-kv-len)         PAST_KV_LEN="$2";           shift 2 ;;
    --image-size)          IMAGE_SIZE="$2";             shift 2 ;;
    --warmup)              WARMUP="$2";                 shift 2 ;;
    --iterations)          ITERATIONS="$2";             shift 2 ;;
    --inference-warmup)    INFERENCE_WARMUP="$2";       shift 2 ;;
    --max-generate-length) MAX_GENERATE_LENGTH="$2";    shift 2 ;;
    --input-vlm-json)      INPUT_VLM_JSON="$2";         shift 2 ;;
    --skip-prefill)        SKIP_PREFILL=true;           shift   ;;
    --skip-decode)         SKIP_DECODE=true;            shift   ;;
    --skip-visual)         SKIP_VISUAL=true;            shift   ;;
    --skip-profile)        SKIP_PROFILE=true;           shift   ;;
    --quick)
      # Smoke-test parameters — faster iteration, NOT the NVIDIA published workload.
      INPUT_LEN=128; PAST_KV_LEN=128; IMAGE_SIZE="320x320"
      WARMUP=1; ITERATIONS=3; INFERENCE_WARMUP=3
      shift ;;
    --dry-run)             DRY_RUN=true;                shift   ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── environment validation ────────────────────────────────────────────────────

: "${TENSORRT_EDGE_LLM_ROOT:?Set TENSORRT_EDGE_LLM_ROOT (source scripts/edge_vlm_env.sh)}"
: "${EDGE_VLM_LLM_ENGINE_DIR:?Set EDGE_VLM_LLM_ENGINE_DIR (source scripts/edge_vlm_env.sh)}"
: "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:?Set EDGE_VLM_MULTIMODAL_ENGINE_DIR (source scripts/edge_vlm_env.sh)}"
: "${EDGELLM_PLUGIN_PATH:?Set EDGELLM_PLUGIN_PATH (source scripts/edge_vlm_env.sh)}"

LLM_BENCH="${TENSORRT_EDGE_LLM_ROOT}/build/examples/llm/llm_bench"
LLM_INFERENCE="${TENSORRT_EDGE_LLM_ROOT}/build/examples/llm/llm_inference"

resolve_engine_provenance() {
  local resolved
  resolved="$(
    python3 "${SCRIPT_DIR}/benchmark_metadata.py" \
      --llm-engine-dir "${EDGE_VLM_LLM_ENGINE_DIR:-}" \
      --multimodal-engine-dir "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}" \
      --model-name "${EDGE_VLM_MODEL_NAME:-}" \
      --engine-profile-id "${EDGE_VLM_ENGINE_PROFILE_ID:-}" \
      --output-provenance-lines
  )"
  mapfile -t _resolved_lines <<< "${resolved}"
  RESOLVED_MODEL_NAME="${_resolved_lines[0]:-${EDGE_VLM_MODEL_NAME:-}}"
  RESOLVED_ENGINE_PROFILE_ID="${_resolved_lines[1]:-${EDGE_VLM_ENGINE_PROFILE_ID:-}}"
  RESOLVED_LLM_ENGINE_DIR="${_resolved_lines[2]:-${EDGE_VLM_LLM_ENGINE_DIR:-}}"
  RESOLVED_MULTIMODAL_ENGINE_DIR="${_resolved_lines[3]:-${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}}"
  ENGINE_PROVENANCE_JSON="${_resolved_lines[4]:-\{\}}"
}

canonical_path() {
  python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' "$1"
}

resolve_engine_provenance

EXPECTED_VISUAL_ENGINE_DIR="$(canonical_path "${RESOLVED_MULTIMODAL_ENGINE_DIR}/visual")"
VISUAL_ENGINE_DIR="${EXPECTED_VISUAL_ENGINE_DIR}"
if [[ -n "${EDGE_VLM_VISUAL_ENGINE_DIR:-}" ]]; then
  OVERRIDE_VISUAL_ENGINE_DIR="$(canonical_path "${EDGE_VLM_VISUAL_ENGINE_DIR}")"
  if [[ "${OVERRIDE_VISUAL_ENGINE_DIR}" != "${EXPECTED_VISUAL_ENGINE_DIR}" ]]; then
    echo "ERROR: EDGE_VLM_VISUAL_ENGINE_DIR must resolve to ${EXPECTED_VISUAL_ENGINE_DIR}" >&2
    echo "       Refusing to benchmark visual.engine from ${OVERRIDE_VISUAL_ENGINE_DIR} while provenance records ${RESOLVED_MULTIMODAL_ENGINE_DIR}." >&2
    exit 1
  fi
  VISUAL_ENGINE_DIR="${OVERRIDE_VISUAL_ENGINE_DIR}"
fi

if [[ "${DRY_RUN}" == "false" ]]; then
  if [[ ! -x "${LLM_BENCH}" ]]; then
    echo "ERROR: llm_bench not found or not executable: ${LLM_BENCH}" >&2
    echo "       Build TensorRT Edge-LLM first." >&2
    exit 1
  fi
  if [[ ! -x "${LLM_INFERENCE}" ]]; then
    echo "ERROR: llm_inference not found or not executable: ${LLM_INFERENCE}" >&2
    exit 1
  fi
  if [[ "${SKIP_VISUAL}" == "false" && ! -f "${VISUAL_ENGINE_DIR}/visual.engine" ]]; then
    echo "ERROR: visual.engine not found: ${VISUAL_ENGINE_DIR}/visual.engine" >&2
    echo "       Set EDGE_VLM_VISUAL_ENGINE_DIR to the visual engine directory." >&2
    exit 1
  fi
fi

# ── output directory ──────────────────────────────────────────────────────────

if [[ "${DRY_RUN}" == "false" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  echo "Benchmark artifacts: ${OUTPUT_DIR}"
else
  echo "[DRY RUN] Would write to: ${OUTPUT_DIR}"
fi

# ── state tracking ────────────────────────────────────────────────────────────

ERRORS=()
SKIPPED_MODES=()

PREFILL_OUT=""
DECODE_OUT=""
VISUAL_OUT=""
PROFILE_OUT=""
INFERENCE_OUT=""

# ── metadata collection ───────────────────────────────────────────────────────

collect_metadata() {
  BASE_METADATA_JSON="$(
  python3 "${SCRIPT_DIR}/benchmark_metadata.py" \
    --llm-engine-dir "${RESOLVED_LLM_ENGINE_DIR}" \
    --multimodal-engine-dir "${RESOLVED_MULTIMODAL_ENGINE_DIR}" \
    --model-name "${RESOLVED_MODEL_NAME}" \
    --engine-profile-id "${RESOLVED_ENGINE_PROFILE_ID}" \
    --edge-llm-root "${TENSORRT_EDGE_LLM_ROOT}" \
    --output -
  )"

  # Pass all values via environment variables to avoid shell interpolation inside
  # Python source code.  The single-quoted heredoc prevents any expansion.
  _BASE_METADATA_JSON="${BASE_METADATA_JSON}" \
  BATCH_SIZE_V="${BATCH_SIZE}" \
  INPUT_LEN_V="${INPUT_LEN}" \
  PAST_KV_LEN_V="${PAST_KV_LEN}" \
  IMAGE_SIZE_V="${IMAGE_SIZE}" \
  WARMUP_V="${WARMUP}" \
  ITERATIONS_V="${ITERATIONS}" \
  INFERENCE_WARMUP_V="${INFERENCE_WARMUP}" \
  MAX_GENERATE_LENGTH_V="${MAX_GENERATE_LENGTH}" \
  INPUT_VLM_JSON_V="${INPUT_VLM_JSON}" \
  python3 <<'PYEOF'
import json, os

def _n(v):
    return v if v else None

def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0

metadata = json.loads(os.environ.get("_BASE_METADATA_JSON", "{}"))
metadata.update({
    "batch_size": _i(os.environ.get("BATCH_SIZE_V")),
    "input_len": _i(os.environ.get("INPUT_LEN_V")),
    "past_kv_len": _i(os.environ.get("PAST_KV_LEN_V")),
    "image_size": os.environ.get("IMAGE_SIZE_V", ""),
    "max_generate_length": _i(os.environ.get("MAX_GENERATE_LENGTH_V")),
    "warmup_iterations": _i(os.environ.get("WARMUP_V")),
    "measured_iterations": _i(os.environ.get("ITERATIONS_V")),
    "inference_warmup_runs": _i(os.environ.get("INFERENCE_WARMUP_V")),
    "input_vlm_json": _n(os.environ.get("INPUT_VLM_JSON_V")),
})
print(json.dumps(metadata, indent=2, sort_keys=True))
PYEOF
}

# ── prefill ───────────────────────────────────────────────────────────────────

if [[ "${SKIP_PREFILL}" == "true" ]]; then
  SKIPPED_MODES+=("prefill")
else
  PREFILL_CMD=(
    "${LLM_BENCH}"
    --mode prefill
    --engineDir "${RESOLVED_LLM_ENGINE_DIR}"
    --batchSize "${BATCH_SIZE}"
    --inputLen "${INPUT_LEN}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --profile
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] ${PREFILL_CMD[*]}"
    PREFILL_OUT="llm_bench_prefill.txt"
  else
    PREFILL_OUT="${OUTPUT_DIR}/llm_bench_prefill.txt"
    echo "==> llm_bench --mode prefill"
    echo "    ${PREFILL_CMD[*]}"
    if "${PREFILL_CMD[@]}" > "${PREFILL_OUT}" 2>&1; then
      echo "OK: prefill benchmark written to ${PREFILL_OUT}"
    else
      echo "ERROR: prefill benchmark failed" >&2
      ERRORS+=("prefill failed")
      PREFILL_OUT=""
    fi
  fi
fi

# ── decode ────────────────────────────────────────────────────────────────────

if [[ "${SKIP_DECODE}" == "true" ]]; then
  SKIPPED_MODES+=("decode")
else
  DECODE_CMD=(
    "${LLM_BENCH}"
    --mode decode
    --engineDir "${RESOLVED_LLM_ENGINE_DIR}"
    --batchSize "${BATCH_SIZE}"
    --pastKVLen "${PAST_KV_LEN}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --profile
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] ${DECODE_CMD[*]}"
    DECODE_OUT="llm_bench_decode.txt"
  else
    DECODE_OUT="${OUTPUT_DIR}/llm_bench_decode.txt"
    echo "==> llm_bench --mode decode"
    echo "    ${DECODE_CMD[*]}"
    if "${DECODE_CMD[@]}" > "${DECODE_OUT}" 2>&1; then
      echo "OK: decode benchmark written to ${DECODE_OUT}"
    else
      echo "ERROR: decode benchmark failed" >&2
      ERRORS+=("decode failed")
      DECODE_OUT=""
    fi
  fi
fi

# ── visual ────────────────────────────────────────────────────────────────────

if [[ "${SKIP_VISUAL}" == "true" ]]; then
  SKIPPED_MODES+=("visual")
else
  VISUAL_CMD=(
    "${LLM_BENCH}"
    --mode visual
    --engineDir "${VISUAL_ENGINE_DIR}"
    --imageSize "${IMAGE_SIZE}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
    --profile
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] ${VISUAL_CMD[*]}"
    VISUAL_OUT="llm_bench_visual.txt"
  else
    VISUAL_OUT="${OUTPUT_DIR}/llm_bench_visual.txt"
    echo "==> llm_bench --mode visual"
    echo "    ${VISUAL_CMD[*]}"
    if "${VISUAL_CMD[@]}" > "${VISUAL_OUT}" 2>&1; then
      echo "OK: visual benchmark written to ${VISUAL_OUT}"
    else
      echo "ERROR: visual benchmark failed" >&2
      if grep -qF "Image data must be 4D [T, H, W, C]" "${VISUAL_OUT}"; then
        echo "       The pinned TensorRT Edge-LLM llm_bench constructs a 3D dummy image." >&2
        echo "       Apply the documented Qwen3-VL visual benchmark workaround in docs/benchmarking.md." >&2
      elif grep -qF "not divisible by patchSize * mergeSize" "${VISUAL_OUT}"; then
        echo "       Choose an HxW image size divisible by patch_size * spatial_merge_size." >&2
      fi
      ERRORS+=("visual failed")
      VISUAL_OUT=""
    fi
  fi
fi

# ── end-to-end profiling with llm_inference --dumpProfile ────────────────────

if [[ "${SKIP_PROFILE}" == "true" ]]; then
  SKIPPED_MODES+=("profile")
elif [[ -z "${INPUT_VLM_JSON}" ]]; then
  echo "WARNING: --input-vlm-json not set; skipping llm_inference --dumpProfile" >&2
  SKIPPED_MODES+=("profile")
else
  PROFILE_OUT="${OUTPUT_DIR}/llm_inference_profile.json"
  INFERENCE_OUT="${OUTPUT_DIR}/llm_inference_output.json"

  PROFILE_CMD=(
    "${LLM_INFERENCE}"
    --engineDir "${RESOLVED_LLM_ENGINE_DIR}"
    --multimodalEngineDir "${RESOLVED_MULTIMODAL_ENGINE_DIR}"
    --inputFile "${INPUT_VLM_JSON}"
    --outputFile "${INFERENCE_OUT}"
    --maxGenerateLength "${MAX_GENERATE_LENGTH}"
    --warmup "${INFERENCE_WARMUP}"
    --dumpProfile
    --profileOutputFile "${PROFILE_OUT}"
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] ${PROFILE_CMD[*]}"
  else
    echo "==> llm_inference --dumpProfile"
    echo "    ${PROFILE_CMD[*]}"
    if "${PROFILE_CMD[@]}"; then
      echo "OK: profile written to ${PROFILE_OUT}"
    else
      echo "ERROR: llm_inference profiling failed" >&2
      ERRORS+=("profile failed")
      PROFILE_OUT=""
      INFERENCE_OUT=""
    fi
  fi
fi

# ── write manifest JSON ───────────────────────────────────────────────────────

RUN_ID="native_bench_${TIMESTAMP}"
RECORDED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

manifest_file="${OUTPUT_DIR}/manifest.json"

if [[ "${DRY_RUN}" == "false" ]]; then
  METADATA_JSON=$(collect_metadata 2>/dev/null || echo '{}')

  # Encode arrays as SOH-delimited strings; the single-quoted heredoc prevents
  # any shell expansion inside the Python source, avoiding injection risks.
  _SKIPPED="$(IFS=$'\x01'; echo "${SKIPPED_MODES[*]+"${SKIPPED_MODES[*]}"}")" \
  _ERRORS="$(IFS=$'\x01'; echo "${ERRORS[*]+"${ERRORS[*]}"}")" \
  _PREFILL_OUT="${PREFILL_OUT}" \
  _DECODE_OUT="${DECODE_OUT}" \
  _VISUAL_OUT="${VISUAL_OUT}" \
  _PROFILE_OUT="${PROFILE_OUT}" \
  _INFERENCE_OUT="${INFERENCE_OUT}" \
  _RUN_ID="${RUN_ID}" \
  _RECORDED_AT="${RECORDED_AT}" \
  _METADATA_JSON="${METADATA_JSON}" \
  python3 - "${manifest_file}" <<'PYEOF'
import json, os, sys

_SEP = "\x01"

def _nullable_basename(v):
    return os.path.basename(v) if v else None

def _split_env(key):
    raw = os.environ.get(key, "")
    return [item for item in raw.split(_SEP) if item]

skipped_modes = _split_env("_SKIPPED")
errors = _split_env("_ERRORS")

metadata = {}
try:
    metadata = json.loads(os.environ.get("_METADATA_JSON", "{}"))
except json.JSONDecodeError:
    pass

manifest = {
    "schema_version": "1",
    "run_id": os.environ["_RUN_ID"],
    "recorded_at": os.environ["_RECORDED_AT"],
    "metadata": metadata,
    "engine_provenance": metadata.get("engine_provenance"),
    "llm_bench_prefill": _nullable_basename(os.environ.get("_PREFILL_OUT")),
    "llm_bench_decode": _nullable_basename(os.environ.get("_DECODE_OUT")),
    "llm_bench_visual": _nullable_basename(os.environ.get("_VISUAL_OUT")),
    "llm_inference_profile": _nullable_basename(os.environ.get("_PROFILE_OUT")),
    "llm_inference_output": _nullable_basename(os.environ.get("_INFERENCE_OUT")),
    "skipped_modes": skipped_modes,
    "errors": errors,
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

# ── exit nonzero if any requested benchmark failed ────────────────────────────

if [[ "${#ERRORS[@]}" -gt 0 ]]; then
  echo "ERROR: ${#ERRORS[@]} benchmark(s) failed: ${ERRORS[*]}" >&2
  exit 1
fi
