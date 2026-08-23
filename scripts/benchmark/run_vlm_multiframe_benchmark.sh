#!/usr/bin/env bash
# run_vlm_multiframe_benchmark.sh
#
# VLM multi-frame latency characterization benchmark.
#
# Sweeps frame-count conditions (F1, F2, F4, F8) using a user-supplied
# sequence of ordered image frames, holding model, engines, precision,
# prompt text, max output tokens, image resolution, power mode, and clocks
# constant across all conditions.
#
# NOTE on lifecycle semantics
# ---------------------------
#   direct : launches a fresh llm_inference process per inference — includes
#            cold-start (process init, engine load, tokenizer init).
#            cold_start_total_ms is recorded separately from model/runtime
#            stage timings.
#
# IPC path is NOT included in this benchmark because vlm_single_shot_client
# accepts a single --image argument and cannot submit a multi-image request.
# Extend the IPC client separately before adding multi-frame IPC support.
#
# This script does NOT:
#   - Change power mode, clock caps, model size, quantization, or engines.
#   - Fabricate TTFT, decode time, or visual token count when the runtime
#     does not expose them; those fields are written as null.
#   - Modify the existing single-frame latency benchmark
#     (run_vlm_latency_benchmark.sh) or Thor contention benchmark
#     (run_thor_pipeline_benchmarks.sh).
#
# Prompt policy
# -------------
# A compact temporal prompt is used that requests one short structured
# result for the entire frame set — equivalent to the successful B/C
# compact_odd_json condition from the single-frame benchmark.
#
# Input request shape
# -------------------
# Uses the Thor-validated TensorRT Edge-LLM VLM request shape:
#   requests -> messages -> content[]   (role: user)
# Multiple image content items {"type":"image","image":path} are placed in
# temporal order, followed by one text item.  SHA-256 content hashes are
# stored in JSONL benchmark metadata (frame_paths field), not in the model
# message payload.
#
# Native profiling
# ----------------
# Uses the same validated CLI contract as run_vlm_latency_benchmark.sh:
#   --engineDir  --multimodalEngineDir  --maxGenerateLength
#   --inputFile  --outputFile  --warmup 0
#   --dumpProfile  --profileOutputFile
#
# Profile fields parsed (exact Thor profile schema from PR #64):
#   multimodal.total_image_tokens                        — visual token count
#   stages[stage_id=vision_encoder].average_time_per_run_ms
#   prefill.average_time_per_run_ms
#   generation.generated_tokens / tokens_per_second / total_time_ms
#   stages[stage_id=llm_generation].total_gpu_time_ms
# TTFT remains null unless explicitly emitted by the runtime.
#
# Artifact naming
# ---------------
# Warmup and measured iterations use distinct sub-directories:
#   warmup_iter_N  / measured_iter_N
# so that artifacts from different phases never collide.
#
# Required environment (source scripts/edge_vlm_env.sh before running):
#   TENSORRT_EDGE_LLM_ROOT         root of TensorRT Edge-LLM checkout
#   EDGE_VLM_LLM_ENGINE_DIR        path to LLM engine directory
#   EDGE_VLM_MULTIMODAL_ENGINE_DIR path to multimodal engine directory
#   EDGE_VLM_MODEL_NAME            model identifier string (for metadata)
#
# Usage
# -----
#   source scripts/edge_vlm_env.sh
#   bash scripts/benchmark/run_vlm_multiframe_benchmark.sh \
#     --sequence-dir /absolute/path/to/ordered_frames \
#     --frame-counts 1,2,4,8 \
#     --max-output-tokens 32 \
#     --warmup 1 \
#     --iterations 3 \
#     --output-dir /tmp/vlm_multiframe_bench
#
# Options
#   --sequence-dir DIR       Directory containing ordered image frames (required)
#   --frame-counts LIST      Comma-separated frame counts (default: 1,2,4,8)
#   --max-output-tokens N    Maximum output tokens (default: 32)
#   --output-dir DIR         Directory for all output artifacts
#                            (default: /tmp/vlm_multiframe_bench_TIMESTAMP)
#   --warmup N               Warmup iterations per condition (default: 1)
#   --iterations N           Measured iterations per condition (default: 3)
#   --dry-run                Print commands without executing them

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/tmp/vlm_multiframe_bench_${TIMESTAMP}"
FRAME_COUNTS="1,2,4,8"
MAX_OUTPUT_TOKENS=32
WARMUP=1
ITERATIONS=3
SEQUENCE_DIR=""
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence-dir)     SEQUENCE_DIR="$2";         shift 2 ;;
        --frame-counts)     FRAME_COUNTS="$2";         shift 2 ;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2";   shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";           shift 2 ;;
        --warmup)           WARMUP="$2";               shift 2 ;;
        --iterations)       ITERATIONS="$2";           shift 2 ;;
        --dry-run)          DRY_RUN=true;              shift ;;
        *) echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done

# ── validate required arguments ───────────────────────────────────────────────

if [[ -z "${SEQUENCE_DIR}" ]]; then
    echo "ERROR: --sequence-dir is required" >&2
    echo "       Provide a directory containing ordered JPEG/PNG image frames." >&2
    exit 1
fi

if [[ ! -d "${SEQUENCE_DIR}" ]]; then
    echo "ERROR: sequence directory does not exist: ${SEQUENCE_DIR}" >&2
    exit 1
fi

# ── derived paths ─────────────────────────────────────────────────────────────

JSONL_OUT="${OUTPUT_DIR}/vlm_multiframe_${TIMESTAMP}.jsonl"
REPORT_JSON="${OUTPUT_DIR}/vlm_multiframe_report.json"
REPORT_TXT="${OUTPUT_DIR}/vlm_multiframe_report.txt"

# ── prompt policy ─────────────────────────────────────────────────────────────

PROMPT_TEXT="You are an autonomous robot perception system. Analyze this temporal sequence of images and describe the scene as compact JSON with keys: objects, actions, hazards, navigable. Be concise. Provide one result for the full sequence."

# ── utilities ─────────────────────────────────────────────────────────────────

_run() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

_now_iso() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

_sha256_file() {
    # Return full SHA-256 hex of a file.
    sha256sum "$1" | awk '{print $1}'
}

_sha256_string() {
    # Return first 12 hex chars of SHA-256 of a string.
    printf '%s' "$1" | sha256sum | cut -c1-12
}

_write_record() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] write_record"
        return
    fi
    local json="$1"
    echo "${json}" >> "${JSONL_OUT}"
}

_validate_image() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: image not found: ${path}" >&2; return 1
    fi
    local size
    size=$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}" 2>/dev/null || echo 0)
    if [[ "${size}" -eq 0 ]]; then
        echo "ERROR: image is zero bytes: ${path}" >&2; return 1
    fi
    local magic
    magic=$(xxd -p -l 4 "${path}" 2>/dev/null || od -A n -N 4 -t x1 "${path}" 2>/dev/null | tr -d ' \n')
    case "${magic,,}" in
        ffd8ff*)   ;;  # JPEG
        89504e47*) ;;  # PNG
        *)
            echo "ERROR: ${path}: not a valid JPEG or PNG (magic=${magic})" >&2
            return 1
            ;;
    esac
    return 0
}

# ── load ordered frame sequence ───────────────────────────────────────────────

_load_sequence() {
    mapfile -t _ALL_FRAMES < <(
        find "${SEQUENCE_DIR}" -maxdepth 1 -type f \
            \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort
    )

    SEQUENCE_FRAMES=()
    for img in "${_ALL_FRAMES[@]}"; do
        if _validate_image "${img}"; then
            SEQUENCE_FRAMES+=("${img}")
        else
            echo "WARNING: excluding invalid image from sequence: ${img}" >&2
        fi
    done

    if [[ ${#SEQUENCE_FRAMES[@]} -eq 0 ]]; then
        echo "ERROR: no valid images found in ${SEQUENCE_DIR}" >&2
        exit 1
    fi

    echo "[setup] Sequence directory: ${SEQUENCE_DIR}"
    echo "[setup] Valid frames found: ${#SEQUENCE_FRAMES[@]}"
    for img in "${SEQUENCE_FRAMES[@]}"; do
        echo "  ${img}"
    done
}

# ── frame selection ───────────────────────────────────────────────────────────

_select_frames() {
    # Select N evenly-spaced frames from SEQUENCE_FRAMES (preserving temporal order).
    # Populates SELECTED_FRAMES array.
    local n="$1"
    local total=${#SEQUENCE_FRAMES[@]}

    if [[ ${total} -lt ${n} ]]; then
        echo "ERROR: insufficient frames for F${n}: need ${n}, have ${total}." >&2
        echo "       Provide a sequence directory with at least ${n} valid frames." >&2
        exit 1
    fi

    SELECTED_FRAMES=()
    if [[ ${n} -eq 1 ]]; then
        SELECTED_FRAMES=("${SEQUENCE_FRAMES[0]}")
        return
    fi
    if [[ ${n} -eq ${total} ]]; then
        SELECTED_FRAMES=("${SEQUENCE_FRAMES[@]}")
        return
    fi

    # Evenly-spaced indices, always including index 0 and index total-1.
    local i
    for (( i=0; i<n; i++ )); do
        local idx
        # Use awk for floating-point arithmetic in bash
        idx=$(awk -v i="${i}" -v n="${n}" -v total="${total}" \
            'BEGIN { printf "%d", int(i * (total - 1) / (n - 1) + 0.5) }')
        SELECTED_FRAMES+=("${SEQUENCE_FRAMES[${idx}]}")
    done
}

# ── request JSON construction ─────────────────────────────────────────────────

_build_request_json() {
    # Build the Thor-validated TensorRT Edge-LLM VLM request JSON for multiple frames.
    # Shape: requests -> messages -> content[]  (role: user)
    # Image items: {"type":"image","image":path} in temporal order, then text prompt.
    # max_output_tokens is NOT embedded in the payload; it is passed via --maxGenerateLength.
    # SHA-256 content hashes go in JSONL metadata (frame_paths), not in the model payload.
    local frames=("$@")

    # Build content array: image items in temporal order, then text prompt.
    local content_items=""
    local sep=""
    for img_path in "${frames[@]}"; do
        content_items="${content_items}${sep}{\"type\":\"image\",\"image\":\"${img_path}\"}"
        sep=","
    done
    # Append text prompt
    local escaped_prompt
    escaped_prompt=$(printf '%s' "${PROMPT_TEXT}" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")
    content_items="${content_items},{\"type\":\"text\",\"text\":${escaped_prompt}}"

    printf '{"requests":[{"messages":[{"role":"user","content":[%s]}]}]}' \
        "${content_items}"
}

# ── direct (native) inference ─────────────────────────────────────────────────

_run_direct_inference() {
    local frame_condition="$1"
    local frame_count="$2"
    local phase="$3"        # "warmup" or "measured"
    local iter_idx="$4"
    local request_json_path="$5"
    local response_path="$6"
    local profile_path="$7"

    local llm_bin="${TENSORRT_EDGE_LLM_ROOT}/build/examples/llm/llm_inference"

    if [[ ! -x "${llm_bin}" ]]; then
        echo "WARNING: llm_inference binary not found: ${llm_bin}; skipping direct run" >&2
        return 1
    fi

    local t_start t_end cold_start_ms
    t_start=$(date +%s%3N)

    local exit_code=0
    _run "${llm_bin}" \
        --engineDir "${EDGE_VLM_LLM_ENGINE_DIR:-}" \
        --multimodalEngineDir "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}" \
        --maxGenerateLength "${MAX_OUTPUT_TOKENS}" \
        --inputFile "${request_json_path}" \
        --outputFile "${response_path}" \
        --warmup 0 \
        --dumpProfile \
        --profileOutputFile "${profile_path}" \
        2>"/tmp/vlm_multiframe_stderr_${TIMESTAMP}_${frame_condition}_${phase}_${iter_idx}.log" \
        || exit_code=$?

    t_end=$(date +%s%3N)
    cold_start_ms=$(( t_end - t_start ))

    if [[ ${exit_code} -ne 0 ]]; then
        echo "llm_inference exited with code ${exit_code}" >&2
        return 1
    fi

    echo "${cold_start_ms}"
}

_parse_native_profile() {
    # Parse NVIDIA Thor profile JSON using the exact schema established by PR #64.
    # Stages are in stages[] keyed by stage_id; prefill uses average_time_per_run_ms.
    # Outputs a JSON fragment with only the fields present in the profile.
    local profile_path="$1"
    if [[ ! -f "${profile_path}" ]]; then
        echo "null"
        return
    fi
    python3 - "${profile_path}" <<'PYEOF'
import json, sys

path = sys.argv[1]
try:
    with open(path) as f:
        p = json.load(f)
except Exception:
    print("null")
    sys.exit(0)

out = {}

# Visual token count (multimodal.total_image_tokens)
mm = p.get("multimodal") or {}
if "total_image_tokens" in mm:
    out["total_image_tokens"] = mm["total_image_tokens"]

# Vision encoder and LLM generation timings from stages[] (Thor PR #64 schema).
# stages[stage_id='vision_encoder'].average_time_per_run_ms -> vision_encoder_ms
# stages[stage_id='llm_generation'].total_gpu_time_ms       -> llm_generation_total_gpu_time_ms
stages = p.get("stages") if isinstance(p, dict) else None
if isinstance(stages, list):
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        sid = stage.get("stage_id")
        if sid == "vision_encoder":
            v = stage.get("average_time_per_run_ms")
            if v is not None:
                out["vision_encoder_ms"] = v
        elif sid == "llm_generation":
            v = stage.get("total_gpu_time_ms")
            if v is not None:
                out["llm_generation_total_gpu_time_ms"] = v

# Prefill: prefill.average_time_per_run_ms
prefill = p.get("prefill") if isinstance(p, dict) else None
if isinstance(prefill, dict):
    v = prefill.get("average_time_per_run_ms")
    if v is not None:
        out["prefill_ms"] = v

# Generation metrics
gen = p.get("generation") or {}
if "generated_tokens" in gen:
    out["actual_output_tokens"] = gen["generated_tokens"]
if "tokens_per_second" in gen:
    out["decode_tokens_per_sec"] = gen["tokens_per_second"]
if "average_time_per_token_ms" in gen:
    out["average_time_per_token_ms"] = gen["average_time_per_token_ms"]
if "total_time_ms" in gen:
    out["decode_ms"] = gen["total_time_ms"]

# Finish reason
fr = p.get("finish_reason") or gen.get("finish_reason")
if fr is not None:
    out["finish_reason"] = fr

print(json.dumps(out))
PYEOF
}

_parse_native_response() {
    # Parse native response JSON; emit actual_output_tokens and finish_reason
    # only when the response contains them.
    local response_path="$1"
    if [[ ! -f "${response_path}" ]]; then
        echo "null"
        return
    fi
    python3 - "${response_path}" <<'PYEOF'
import json, sys

path = sys.argv[1]
try:
    with open(path) as f:
        r = json.load(f)
except Exception:
    print("null")
    sys.exit(0)

out = {}

# Thor llm_inference response: {"responses": [{...}]}
entry = r
responses_list = r.get("responses") if isinstance(r, dict) else None
if isinstance(responses_list, list) and responses_list:
    entry = responses_list[0]
if isinstance(entry, dict):
    text = entry.get("outputText") or entry.get("output_text") or entry.get("text")
    if text:
        out["output_text"] = text
    fr = entry.get("finishReason") or entry.get("finish_reason")
    if fr:
        out["finish_reason"] = fr
    toks_count = entry.get("numOutputTokens")
    if toks_count is None:
        toks_list = entry.get("outputTokens") or entry.get("output_tokens")
        if isinstance(toks_list, list):
            toks_count = len(toks_list)
        elif isinstance(toks_list, int):
            toks_count = toks_list
    if toks_count is not None:
        out["actual_output_tokens"] = toks_count

print(json.dumps(out))
PYEOF
}

# ── inference loop ────────────────────────────────────────────────────────────

_run_condition() {
    local frame_condition="$1"
    local frame_count="$2"

    echo ""
    echo "=== Condition ${frame_condition} | path=direct | frames=${frame_count} ==="

    _select_frames "${frame_count}"
    local frames=("${SELECTED_FRAMES[@]}")

    echo "[${frame_condition}/direct] Selected frames (temporal order):"
    for img in "${frames[@]}"; do
        echo "  ${img}"
    done

    local frame_hashes_json="["
    local sep=""
    for img in "${frames[@]}"; do
        local h
        h=$(_sha256_file "${img}")
        frame_hashes_json="${frame_hashes_json}${sep}{\"path\":\"${img}\",\"sha256\":\"${h}\"}"
        sep=","
    done
    frame_hashes_json="${frame_hashes_json}]"

    local prompt_hash_val
    prompt_hash_val=$(_sha256_string "${PROMPT_TEXT}")

    local total_iters=$(( WARMUP + ITERATIONS ))
    for (( iter=0; iter<total_iters; iter++ )); do
        local is_warmup=false
        local phase="measured"
        local iter_idx
        if [[ ${iter} -lt ${WARMUP} ]]; then
            is_warmup=true
            phase="warmup"
            iter_idx=${iter}
            echo "[${frame_condition}/direct] warmup ${iter_idx}"
        else
            iter_idx=$(( iter - WARMUP ))
            echo "[${frame_condition}/direct] iteration ${iter_idx}"
        fi

        # Use phase in artifact path to prevent warmup/measured collisions.
        local artifact_dir="${OUTPUT_DIR}/artifacts/${frame_condition}/direct/${phase}_iter_${iter_idx}"
        _run mkdir -p "${artifact_dir}"

        local request_json_path="${artifact_dir}/request.json"
        local recorded_at
        recorded_at=$(_now_iso)

        # Build request JSON (Thor-validated shape, no max_output_tokens in payload)
        if [[ "${DRY_RUN}" != "true" ]]; then
            _build_request_json "${frames[@]}" > "${request_json_path}"
        fi

        local response_path="${artifact_dir}/response.json"
        local profile_path="${artifact_dir}/profile.json"

        local cold_start_ms=0
        local success=true
        local error_msg="null"

        if [[ "${DRY_RUN}" != "true" ]]; then
            cold_start_ms=$(_run_direct_inference \
                "${frame_condition}" "${frame_count}" "${phase}" "${iter_idx}" \
                "${request_json_path}" "${response_path}" "${profile_path}") || {
                success=false
                error_msg="\"direct inference failed\""
                cold_start_ms=0
            }
        fi

        local profile_fields="null"
        local response_fields="null"
        if [[ "${success}" == "true" && "${DRY_RUN}" != "true" ]]; then
            profile_fields=$(_parse_native_profile "${profile_path}")
            response_fields=$(_parse_native_response "${response_path}")
        fi

        # Extract parsed fields
        local total_image_tokens vision_encoder_ms prefill_ms decode_ms
        local actual_output_tokens decode_tokens_per_sec llm_gen_gpu_ms finish_reason
        total_image_tokens=$(python3 -c "import json,sys; p=${profile_fields}; print(json.dumps(p.get('total_image_tokens') if p else None))" 2>/dev/null || echo "null")
        vision_encoder_ms=$(python3 -c "import json,sys; p=${profile_fields}; print(json.dumps(p.get('vision_encoder_ms') if p else None))" 2>/dev/null || echo "null")
        prefill_ms=$(python3 -c "import json,sys; p=${profile_fields}; print(json.dumps(p.get('prefill_ms') if p else None))" 2>/dev/null || echo "null")
        decode_ms=$(python3 -c "import json,sys; p=${profile_fields}; print(json.dumps(p.get('decode_ms') if p else None))" 2>/dev/null || echo "null")
        actual_output_tokens=$(python3 -c "import json,sys; p=${profile_fields}; r=${response_fields}; v=(p or {}).get('actual_output_tokens') or (r or {}).get('actual_output_tokens'); print(json.dumps(v))" 2>/dev/null || echo "null")
        decode_tokens_per_sec=$(python3 -c "import json,sys; p=${profile_fields}; print(json.dumps(p.get('decode_tokens_per_sec') if p else None))" 2>/dev/null || echo "null")
        llm_gen_gpu_ms=$(python3 -c "import json,sys; p=${profile_fields}; print(json.dumps(p.get('llm_generation_total_gpu_time_ms') if p else None))" 2>/dev/null || echo "null")
        finish_reason=$(python3 -c "import json,sys; p=${profile_fields}; r=${response_fields}; v=(p or {}).get('finish_reason') or (r or {}).get('finish_reason'); print(json.dumps(v))" 2>/dev/null || echo "null")

        local record
        record=$(python3 -c "
import json
print(json.dumps({
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': '${TIMESTAMP}',
    'recorded_at': '${recorded_at}',
    'frame_condition': '${frame_condition}',
    'frame_count': ${frame_count},
    'path': 'direct',
    'frame_paths': ${frame_hashes_json},
    'prompt_hash': '${prompt_hash_val}',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'actual_output_tokens': ${actual_output_tokens},
    'total_image_tokens': ${total_image_tokens},
    'finish_reason': ${finish_reason},
    'success': $([ '${success}' = 'true' ] && echo 'true' || echo 'false'),
    'error': ${error_msg},
    'cold_start_total_ms': $([ '${DRY_RUN}' != 'true' ] && echo '${cold_start_ms}' || echo 'null'),
    'total_latency_ms': null,
    'ttft_ms': null,
    'vision_encoder_ms': ${vision_encoder_ms},
    'prefill_ms': ${prefill_ms},
    'decode_ms': ${decode_ms},
    'decode_tokens_per_sec': ${decode_tokens_per_sec},
    'llm_generation_total_gpu_time_ms': ${llm_gen_gpu_ms},
    'inference_seconds': null,
    'output_text': null,
    'output_words': null,
    'native_response_path': '${response_path}',
    'native_profile_path': '${profile_path}',
    'ipc_result_path': null,
    'model_name': '${EDGE_VLM_MODEL_NAME:-unknown}',
    'iteration': ${iter_idx},
    'warmup': $([ '${is_warmup}' = 'true' ] && echo 'true' || echo 'false'),
}))")
        _write_record "${record}"
    done
}

# ── main ──────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  VLM Multi-Frame Latency Characterization Benchmark"
echo "  Timestamp: ${TIMESTAMP}"
echo "  Sequence dir: ${SEQUENCE_DIR}"
echo "  Frame counts: ${FRAME_COUNTS}"
echo "  Max output tokens: ${MAX_OUTPUT_TOKENS}"
echo "  Path: direct (IPC requires a multi-image client — not yet available)"
echo "  Warmup: ${WARMUP}  Iterations: ${ITERATIONS}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "================================================================"

_run mkdir -p "${OUTPUT_DIR}"

_load_sequence

# Validate that enough frames exist for the largest requested count
IFS=',' read -ra _FRAME_COUNTS <<< "${FRAME_COUNTS}"

MAX_NEEDED=0
for fc in "${_FRAME_COUNTS[@]}"; do
    if [[ ${fc} -gt ${MAX_NEEDED} ]]; then
        MAX_NEEDED=${fc}
    fi
done

if [[ ${#SEQUENCE_FRAMES[@]} -lt ${MAX_NEEDED} ]]; then
    echo "ERROR: insufficient frames: need ${MAX_NEEDED} for the largest condition," >&2
    echo "       but only ${#SEQUENCE_FRAMES[@]} valid frames found in ${SEQUENCE_DIR}." >&2
    echo "       Provide a sequence directory with at least ${MAX_NEEDED} ordered frames." >&2
    exit 1
fi

echo "[setup] Frame validation passed: ${#SEQUENCE_FRAMES[@]} >= ${MAX_NEEDED} frames needed"

# Run all conditions (direct path only)
for fc in "${_FRAME_COUNTS[@]}"; do
    local_fc_label="F${fc}"
    _run_condition "${local_fc_label}" "${fc}"
done

echo ""
echo "================================================================"
echo "  All conditions complete.  Generating report..."
echo "================================================================"

# Generate report
_run python3 "${SCRIPT_DIR}/vlm_multiframe_report.py" \
    --input "${JSONL_OUT}" \
    --output "${REPORT_JSON}" \
    --text "${REPORT_TXT}"

echo ""
echo "Artifacts:"
echo "  JSONL:  ${JSONL_OUT}"
echo "  JSON:   ${REPORT_JSON}"
echo "  Text:   ${REPORT_TXT}"
echo ""
echo "Done."
