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
#   ipc    : sends all frames in one request to an already-running, warmed
#            edge_vlm_server via IPC socket (vlm_multi_frame_client →
#            edge_vlm_cli → running edge_vlm_server).  This is the persistent-
#            server steady-state path; total_latency_ms is the outer client
#            round-trip.  Stage timings are null; TTFT is null.
#
# Both paths use the same Thor-validated request shape and prompt, so
# direct cold-start and IPC steady-state measurements are directly comparable
# within each frame-count condition.
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
#   EDGE_VLM_WORKER_SOCKET         IPC socket for edge_vlm_server (default: /tmp/edge_vlm.sock)
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
#   --paths LIST             Comma-separated paths: direct,ipc (default: direct,ipc)
#   --sequence-type MODE     images|temporal_images|video (default: images)
#   --fps N                  Optional FPS metadata for temporal modes
#   --frame-timestamps-sec   Optional comma-separated per-frame timestamps (seconds)
#   --render-timestamps      Render visible timestamps on frames (experimental control)
#   --skip-ipc               Alias for --paths direct
#   --skip-direct            Alias for --paths ipc
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
PATHS="direct,ipc"
SEQUENCE_TYPE="images"
FPS=""
FRAME_TIMESTAMPS_SEC=""
RENDER_TIMESTAMPS=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESOLVED_MODEL_NAME="${EDGE_VLM_MODEL_NAME:-unknown}"
RESOLVED_ENGINE_PROFILE_ID="${EDGE_VLM_ENGINE_PROFILE_ID:-}"
RESOLVED_LLM_ENGINE_DIR="${EDGE_VLM_LLM_ENGINE_DIR:-}"
RESOLVED_MULTIMODAL_ENGINE_DIR="${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}"
ENGINE_PROVENANCE_JSON="{}"
IPC_RESOLVED_MODEL_NAME="${EDGE_VLM_MODEL_NAME:-unknown}"
IPC_RESOLVED_ENGINE_PROFILE_ID="${EDGE_VLM_ENGINE_PROFILE_ID:-}"
IPC_RESOLVED_LLM_ENGINE_DIR="${EDGE_VLM_LLM_ENGINE_DIR:-}"
IPC_RESOLVED_MULTIMODAL_ENGINE_DIR="${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}"
IPC_ENGINE_PROVENANCE_JSON="{}"

# ── argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence-dir)     SEQUENCE_DIR="$2";         shift 2 ;;
        --frame-counts)     FRAME_COUNTS="$2";         shift 2 ;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2";   shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";           shift 2 ;;
        --warmup)           WARMUP="$2";               shift 2 ;;
        --iterations)       ITERATIONS="$2";           shift 2 ;;
        --paths)            PATHS="$2";                shift 2 ;;
        --sequence-type)    SEQUENCE_TYPE="$2";        shift 2 ;;
        --fps)              FPS="$2";                  shift 2 ;;
        --frame-timestamps-sec) FRAME_TIMESTAMPS_SEC="$2"; shift 2 ;;
        --render-timestamps) RENDER_TIMESTAMPS=true;   shift ;;
        --skip-ipc)         PATHS="direct";            shift ;;
        --skip-direct)      PATHS="ipc";               shift ;;
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

if [[ "${SEQUENCE_TYPE}" != "images" && "${SEQUENCE_TYPE}" != "temporal_images" && "${SEQUENCE_TYPE}" != "video" ]]; then
    echo "ERROR: --sequence-type must be one of: images, temporal_images, video" >&2
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

_resolve_engine_provenance() {
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
    RESOLVED_MODEL_NAME="${_resolved_lines[0]:-${EDGE_VLM_MODEL_NAME:-unknown}}"
    RESOLVED_ENGINE_PROFILE_ID="${_resolved_lines[1]:-${EDGE_VLM_ENGINE_PROFILE_ID:-}}"
    RESOLVED_LLM_ENGINE_DIR="${_resolved_lines[2]:-${EDGE_VLM_LLM_ENGINE_DIR:-}}"
    RESOLVED_MULTIMODAL_ENGINE_DIR="${_resolved_lines[3]:-${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}}"
    ENGINE_PROVENANCE_JSON="${_resolved_lines[4]:-\{\}}"
}

_resolve_ipc_engine_provenance() {
    local socket_path="${EDGE_VLM_WORKER_SOCKET:-/tmp/edge_vlm.sock}"
    local resolved
    if ! resolved="$(
        python3 "${SCRIPT_DIR}/benchmark_metadata.py" \
            --server-socket-path "${socket_path}" \
            --model-name "${RESOLVED_MODEL_NAME:-${EDGE_VLM_MODEL_NAME:-unknown}}" \
            --engine-profile-id "${RESOLVED_ENGINE_PROFILE_ID:-${EDGE_VLM_ENGINE_PROFILE_ID:-}}" \
            --output-provenance-lines
    )"; then
        echo "ERROR: could not resolve authoritative IPC engine provenance from running edge_vlm_server on ${socket_path}" >&2
        echo "       Start or restart edge_vlm_server with the intended engine paths before benchmarking IPC." >&2
        exit 1
    fi
    mapfile -t _resolved_lines <<< "${resolved}"
    IPC_RESOLVED_MODEL_NAME="${_resolved_lines[0]:-${RESOLVED_MODEL_NAME:-${EDGE_VLM_MODEL_NAME:-unknown}}}"
    IPC_RESOLVED_ENGINE_PROFILE_ID="${_resolved_lines[1]:-${RESOLVED_ENGINE_PROFILE_ID:-${EDGE_VLM_ENGINE_PROFILE_ID:-}}}"
    IPC_RESOLVED_LLM_ENGINE_DIR="${_resolved_lines[2]:-${RESOLVED_LLM_ENGINE_DIR:-${EDGE_VLM_LLM_ENGINE_DIR:-}}}"
    IPC_RESOLVED_MULTIMODAL_ENGINE_DIR="${_resolved_lines[3]:-${RESOLVED_MULTIMODAL_ENGINE_DIR:-${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}}}"
    IPC_ENGINE_PROVENANCE_JSON="${_resolved_lines[4]:-\{\}}"

    if [[ "${IPC_RESOLVED_LLM_ENGINE_DIR}" != "${RESOLVED_LLM_ENGINE_DIR}" \
       || "${IPC_RESOLVED_MULTIMODAL_ENGINE_DIR}" != "${RESOLVED_MULTIMODAL_ENGINE_DIR}" ]]; then
        echo "WARNING: caller-shell engine paths differ from the running edge_vlm_server." >&2
        echo "         Direct records will use caller-shell runtime paths; IPC records will use server-process paths." >&2
        echo "         The generated report will be marked with mixed engine provenance and should be treated as non-comparable." >&2
    fi
}

_runtime_temporal_encoding_for_direct() {
    if [[ "${SEQUENCE_TYPE}" == "images" ]]; then
        echo "ordered_multi_image_no_native_temporal_metadata"
    else
        echo "ordered_multi_image_fallback_no_native_video_fps_timestamp_api_in_pinned_edgellm"
    fi
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
        --engineDir "${RESOLVED_LLM_ENGINE_DIR:-}" \
        --multimodalEngineDir "${RESOLVED_MULTIMODAL_ENGINE_DIR:-}" \
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

# ── IPC path invocation (persistent edge_vlm_server) ─────────────────────────

_run_ipc_inference() {
    # Send all selected frames in a single multi-image IPC request to the
    # already-running edge_vlm_server via vlm_multi_frame_client.
    # Accepts: frame_condition frame_count phase iter_idx result_json_path frames...
    local frame_condition="$1"
    local frame_count="$2"
    local phase="$3"
    local iter_idx="$4"
    local result_json_path="$5"
    shift 5
    local frames=("$@")

    # The ipc path invokes vlm_multi_frame_client via `ros2 run` because the
    # script is installed into the ROS lib directory (lib/edge_vlm_ros/).
    # `ros2` is required only to locate and launch the installed script;
    # vlm_multi_frame_client itself does not depend on ROS at runtime — it
    # communicates with edge_vlm_server directly via edge_vlm_cli using a
    # single kSchemaFlagMultiImage IPC request carrying all frames together.
    # A running edge_vlm_server on EDGE_VLM_WORKER_SOCKET is also required.
    if ! command -v ros2 &>/dev/null; then
        echo "WARNING: ros2 not available — ipc path (vlm_multi_frame_client) skipped for ${frame_condition} ${phase}_iter_${iter_idx}" >&2
        echo "skipped_no_ros2"
        return 0
    fi

    local exit_code=0
    # Build --image arguments for each selected frame.
    local image_args=()
    local img
    for img in "${frames[@]}"; do
        image_args+=(--image "${img}")
    done

    local temporal_args=(--sequence-type "${SEQUENCE_TYPE}")
    if [[ -n "${FPS}" ]]; then
        temporal_args+=(--fps "${FPS}")
    fi
    if [[ -n "${FRAME_TIMESTAMPS_SEC}" ]]; then
        temporal_args+=(--frame-timestamps-sec "${FRAME_TIMESTAMPS_SEC}")
    fi
    if [[ "${RENDER_TIMESTAMPS}" == "true" ]]; then
        temporal_args+=(--render-timestamps)
    fi

    _run ros2 run edge_vlm_ros vlm_multi_frame_client \
        --socket "${EDGE_VLM_WORKER_SOCKET:-/tmp/edge_vlm.sock}" \
        "${image_args[@]}" \
        "${temporal_args[@]}" \
        --prompt "${PROMPT_TEXT}" \
        --max-tokens "${MAX_OUTPUT_TOKENS}" \
        --output "${result_json_path}" \
        --timeout 120 \
        2>"/tmp/vlm_multiframe_ipc_stderr_${TIMESTAMP}_${frame_condition}_${phase}_${iter_idx}.log" \
        || exit_code=$?

    echo "${exit_code}"
}

# ── inference loop ────────────────────────────────────────────────────────────

_run_condition() {
    local frame_condition="$1"
    local frame_count="$2"

    echo ""
    echo "=== Condition ${frame_condition} | frames=${frame_count} | paths=${PATHS} ==="

    _select_frames "${frame_count}"
    local frames=("${SELECTED_FRAMES[@]}")

    echo "[${frame_condition}] Selected frames (temporal order):"
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
    local frame_timestamps_json="null"
    local frame_timestamp_policy='"none"'
    if [[ -n "${FRAME_TIMESTAMPS_SEC}" ]]; then
        frame_timestamps_json=$(
            python3 - "${FRAME_TIMESTAMPS_SEC}" <<'PYEOF'
import json, sys
items = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(json.dumps([float(x) for x in items]))
PYEOF
        )
        frame_timestamp_policy='"explicit"'
    elif [[ -n "${FPS}" && "${SEQUENCE_TYPE}" != "images" ]]; then
        frame_timestamp_policy='"implicit_uniform_from_fps"'
    fi

    # ── direct path ────────────────────────────────────────────────────────────
    if [[ ",${PATHS}," == *",direct,"* ]]; then
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

        # Extract parsed fields — single Python invocation, all data via env vars.
        # profile_fields / response_fields are already JSON objects (or "null")
        # produced by _parse_native_profile / _parse_native_response.  Passing them
        # as environment variables and decoding with json.loads() prevents
        # any shell→Python literal injection.
        local _extracted_fields
        _extracted_fields=$(
            export _BM_PROFILE="${profile_fields}"
            export _BM_RESPONSE="${response_fields}"
            python3 -c "
import json, os
def _jl(s):
    try: return json.loads(s)
    except Exception: return None
p = _jl(os.environ.get('_BM_PROFILE', 'null')) or {}
r = _jl(os.environ.get('_BM_RESPONSE', 'null')) or {}
d = json.dumps
print('\t'.join([
    d(p.get('total_image_tokens')),
    d(p.get('vision_encoder_ms')),
    d(p.get('prefill_ms')),
    d(p.get('decode_ms')),
    d(p.get('actual_output_tokens') if p.get('actual_output_tokens') is not None else r.get('actual_output_tokens')),
    d(p.get('decode_tokens_per_sec')),
    d(p.get('llm_generation_total_gpu_time_ms')),
    d(p.get('finish_reason') if p.get('finish_reason') is not None else r.get('finish_reason')),
]))
" 2>/dev/null || printf 'null\tnull\tnull\tnull\tnull\tnull\tnull\tnull')
        local total_image_tokens vision_encoder_ms prefill_ms decode_ms
        local actual_output_tokens decode_tokens_per_sec llm_gen_gpu_ms finish_reason
        IFS=$'\t' read -r total_image_tokens vision_encoder_ms prefill_ms decode_ms \
            actual_output_tokens decode_tokens_per_sec llm_gen_gpu_ms finish_reason \
            <<< "${_extracted_fields}"

        local record
        # Build the direct record via build_direct_record() in vlm_multiframe_report.py.
        # All dynamic values are passed as _BM_* environment variables so that no
        # shell variable is ever interpolated into Python source (prevents NameError
        # on JSON null/true/false and injection via output_text or path strings).
        record=$(
            export _BM_RUN_ID="${TIMESTAMP}"
            export _BM_RECORDED_AT="${recorded_at}"
            export _BM_FRAME_CONDITION="${frame_condition}"
            export _BM_FRAME_COUNT="${frame_count}"
            export _BM_FRAME_HASHES="${frame_hashes_json}"
            export _BM_PROMPT_HASH="${prompt_hash_val}"
            export _BM_MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS}"
            export _BM_ACTUAL_OUTPUT_TOKENS="${actual_output_tokens}"
            export _BM_TOTAL_IMAGE_TOKENS="${total_image_tokens}"
            export _BM_FINISH_REASON="${finish_reason}"
            export _BM_SUCCESS="$([ "${success}" = 'true' ] && echo 'true' || echo 'false')"
            export _BM_ERROR="${error_msg}"
            export _BM_COLD_START_MS="$([ "${DRY_RUN}" != 'true' ] && echo "${cold_start_ms}" || echo 'null')"
            export _BM_VISION_ENCODER_MS="${vision_encoder_ms}"
            export _BM_PREFILL_MS="${prefill_ms}"
            export _BM_DECODE_MS="${decode_ms}"
            export _BM_DECODE_TOKENS_PER_SEC="${decode_tokens_per_sec}"
            export _BM_LLM_GEN_GPU_MS="${llm_gen_gpu_ms}"
            export _BM_RESPONSE_PATH="${response_path}"
            export _BM_PROFILE_PATH="${profile_path}"
            export _BM_MODEL_NAME="${RESOLVED_MODEL_NAME:-unknown}"
            export _BM_ENGINE_PROVENANCE="${ENGINE_PROVENANCE_JSON}"
            export _BM_SEQUENCE_TYPE="\"${SEQUENCE_TYPE}\""
            export _BM_FPS="$([ -n "${FPS}" ] && echo "${FPS}" || echo 'null')"
            export _BM_FRAME_TIMESTAMPS_SEC="${frame_timestamps_json}"
            export _BM_FRAME_TIMESTAMP_POLICY="${frame_timestamp_policy}"
            export _BM_RENDERED_TIMESTAMPS="$([ "${RENDER_TIMESTAMPS}" = 'true' ] && echo 'true' || echo 'false')"
            export _BM_RUNTIME_TEMPORAL_ENCODING="\"$(_runtime_temporal_encoding_for_direct)\""
            export _BM_TEMPORAL_FALLBACK_USED="$([ "${SEQUENCE_TYPE}" = 'images' ] && echo 'false' || echo 'true')"
            export _BM_ITERATION="${iter_idx}"
            export _BM_IS_WARMUP="$([ "${is_warmup}" = 'true' ] && echo 'true' || echo 'false')"
            PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
            python3 -c "from vlm_multiframe_report import build_direct_record; print(build_direct_record())"
        )
        _write_record "${record}"
        done
    fi

    # ── ipc path ───────────────────────────────────────────────────────────────
    if [[ ",${PATHS}," == *",ipc,"* ]]; then
        local total_iters=$(( WARMUP + ITERATIONS ))
        for (( iter=0; iter<total_iters; iter++ )); do
            local is_warmup=false
            local phase="measured"
            local iter_idx
            if [[ ${iter} -lt ${WARMUP} ]]; then
                is_warmup=true
                phase="warmup"
                iter_idx=${iter}
                echo "[${frame_condition}/ipc] warmup ${iter_idx}"
            else
                iter_idx=$(( iter - WARMUP ))
                echo "[${frame_condition}/ipc] iteration ${iter_idx}"
            fi

            # Use phase in artifact path to prevent warmup/measured collisions.
            local ipc_artifact_dir="${OUTPUT_DIR}/artifacts/${frame_condition}/ipc/${phase}_iter_${iter_idx}"
            _run mkdir -p "${ipc_artifact_dir}"

            local result_json_path="${ipc_artifact_dir}/ipc_result.json"
            local ipc_recorded_at
            ipc_recorded_at=$(_now_iso)

            local t_start t_end total_latency_ms
            t_start=$(date +%s%3N)

            local ipc_exit_code
            if [[ "${DRY_RUN}" != "true" ]]; then
                ipc_exit_code=$(_run_ipc_inference \
                    "${frame_condition}" "${frame_count}" "${phase}" "${iter_idx}" \
                    "${result_json_path}" "${frames[@]}")
            else
                ipc_exit_code=0
                _run echo "[dry-run] vlm_multi_frame_client ${frame_condition} ${phase}_iter_${iter_idx}"
            fi

            t_end=$(date +%s%3N)
            total_latency_ms=$(( t_end - t_start ))

            local ipc_success=true
            local ipc_error_msg="null"
            if [[ "${ipc_exit_code}" == "skipped_no_ros2" ]]; then
                ipc_success=false
                ipc_error_msg='"ros2 not available — ipc path (vlm_multi_frame_client) skipped"'
                total_latency_ms=0
            elif [[ "${ipc_exit_code}" != "0" ]]; then
                ipc_success=false
                ipc_error_msg="\"ipc client exited with code ${ipc_exit_code}\""
            fi

            # Parse IPC result artifact — single python3 invocation to avoid
            # four separate interpreter start-ups and file reads per iteration.
            local ipc_output_text="null"
            local ipc_output_words="null"
            local ipc_inference_seconds="null"
            local ipc_requested_sequence_type="null"
            local ipc_runtime_temporal_encoding="null"
            local ipc_temporal_fallback_used="null"
            local ipc_rendered_timestamps="$([ "${RENDER_TIMESTAMPS}" = 'true' ] && echo 'true' || echo 'false')"
            local ipc_sequence_type_json
            ipc_sequence_type_json="\"${SEQUENCE_TYPE}\""
            local ipc_actual_latency_ms
            if [[ "${ipc_success}" == "true" && "${DRY_RUN}" != "true" && -f "${result_json_path}" ]]; then
                local ipc_fields
                ipc_fields=$(python3 - "${result_json_path}" "${total_latency_ms}" <<'PYEOF'
import json, sys
path, fallback_ms = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        r = json.load(f)
    ot = json.dumps(r.get("output_text"))
    ow = json.dumps(r.get("output_words"))
    iv = r.get("inference_seconds")
    inf = json.dumps(float(iv) if iv is not None else None)
    rq = json.dumps(r.get("requested_sequence_type"))
    enc = json.dumps(r.get("runtime_temporal_encoding"))
    tfb = json.dumps(r.get("temporal_fallback_used"))
    sts = json.dumps(r.get("sequence_type"))
    rts = json.dumps(r.get("rendered_timestamps"))
    cv = r.get("client_latency_ms")
    lms = str(int(cv)) if cv is not None else fallback_ms
except Exception:
    ot, ow, inf, rq, enc, tfb, sts, rts, lms = "null", "null", "null", "null", "null", "null", "null", "null", fallback_ms
print(f"{ot}\t{ow}\t{inf}\t{rq}\t{enc}\t{tfb}\t{sts}\t{rts}\t{lms}")
PYEOF
                2>/dev/null || printf 'null\tnull\tnull\tnull\tnull\tnull\tnull\tnull\t%s' "${total_latency_ms}")
                IFS=$'\t' read -r ipc_output_text ipc_output_words ipc_inference_seconds ipc_requested_sequence_type ipc_runtime_temporal_encoding ipc_temporal_fallback_used ipc_sequence_type_json ipc_rendered_timestamps ipc_actual_latency_ms <<< "${ipc_fields}"
            else
                ipc_actual_latency_ms="${total_latency_ms}"
            fi

            local ipc_result_path_json
            if [[ "${ipc_success}" == "true" && -f "${result_json_path}" ]]; then
                ipc_result_path_json="\"${result_json_path}\""
            else
                ipc_result_path_json="null"
            fi

            local ipc_record
            # Build the IPC record via build_ipc_record() in vlm_multiframe_report.py.
            # All dynamic values are passed as _BM_* env vars — no shell variable is
            # interpolated into Python source.
            ipc_record=$(
                export _BM_RUN_ID="${TIMESTAMP}"
                export _BM_RECORDED_AT="${ipc_recorded_at}"
                export _BM_FRAME_CONDITION="${frame_condition}"
                export _BM_FRAME_COUNT="${frame_count}"
                export _BM_FRAME_HASHES="${frame_hashes_json}"
                export _BM_PROMPT_HASH="${prompt_hash_val}"
                export _BM_MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS}"
                export _BM_SUCCESS="$([ "${ipc_success}" = 'true' ] && echo 'true' || echo 'false')"
                export _BM_ERROR="${ipc_error_msg}"
                export _BM_TOTAL_LATENCY="$([ "${ipc_success}" = 'true' ] && echo "${ipc_actual_latency_ms}" || echo 'null')"
                export _BM_INFERENCE_SECONDS="${ipc_inference_seconds}"
                export _BM_OUTPUT_TEXT="${ipc_output_text}"
                export _BM_OUTPUT_WORDS="${ipc_output_words}"
                export _BM_IPC_RESULT_PATH="${ipc_result_path_json}"
                export _BM_MODEL_NAME="${IPC_RESOLVED_MODEL_NAME:-unknown}"
                export _BM_ENGINE_PROVENANCE="${IPC_ENGINE_PROVENANCE_JSON}"
                export _BM_SEQUENCE_TYPE="${ipc_sequence_type_json}"
                export _BM_FPS="$([ -n "${FPS}" ] && echo "${FPS}" || echo 'null')"
                export _BM_FRAME_TIMESTAMPS_SEC="${frame_timestamps_json}"
                export _BM_FRAME_TIMESTAMP_POLICY="${frame_timestamp_policy}"
                export _BM_RENDERED_TIMESTAMPS="${ipc_rendered_timestamps}"
                export _BM_REQUESTED_SEQUENCE_TYPE="${ipc_requested_sequence_type}"
                export _BM_RUNTIME_TEMPORAL_ENCODING="${ipc_runtime_temporal_encoding}"
                export _BM_TEMPORAL_FALLBACK_USED="${ipc_temporal_fallback_used}"
                export _BM_ITERATION="${iter_idx}"
                export _BM_IS_WARMUP="$([ "${is_warmup}" = 'true' ] && echo 'true' || echo 'false')"
                PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
                python3 -c "from vlm_multiframe_report import build_ipc_record; print(build_ipc_record())"
            )
            _write_record "${ipc_record}"
        done
    fi
}

# ── main ──────────────────────────────────────────────────────────────────────

_resolve_engine_provenance
if [[ ",${PATHS}," == *",ipc,"* && "${DRY_RUN}" != "true" ]]; then
    _resolve_ipc_engine_provenance
fi

echo "================================================================"
echo "  VLM Multi-Frame Latency Characterization Benchmark"
echo "  Timestamp: ${TIMESTAMP}"
echo "  Sequence dir: ${SEQUENCE_DIR}"
echo "  Frame counts: ${FRAME_COUNTS}"
echo "  Max output tokens: ${MAX_OUTPUT_TOKENS}"
echo "  Sequence type: ${SEQUENCE_TYPE}"
echo "  FPS: ${FPS:-unset}"
echo "  Frame timestamps sec: ${FRAME_TIMESTAMPS_SEC:-unset}"
echo "  Render timestamps: ${RENDER_TIMESTAMPS}"
echo "  Paths: ${PATHS}"
echo "  Engine: ${RESOLVED_MODEL_NAME:-unknown} ${RESOLVED_ENGINE_PROFILE_ID:+(${RESOLVED_ENGINE_PROFILE_ID})}"
echo "  LLM engine dir: ${RESOLVED_LLM_ENGINE_DIR:-unset}"
echo "  Multimodal dir: ${RESOLVED_MULTIMODAL_ENGINE_DIR:-unset}"
if [[ ",${PATHS}," == *",ipc,"* && "${DRY_RUN}" != "true" ]]; then
    echo "  IPC server engine: ${IPC_RESOLVED_MODEL_NAME:-unknown} ${IPC_RESOLVED_ENGINE_PROFILE_ID:+(${IPC_RESOLVED_ENGINE_PROFILE_ID})}"
    echo "  IPC server LLM dir: ${IPC_RESOLVED_LLM_ENGINE_DIR:-unset}"
    echo "  IPC server multimodal dir: ${IPC_RESOLVED_MULTIMODAL_ENGINE_DIR:-unset}"
fi
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

# Run all conditions
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
