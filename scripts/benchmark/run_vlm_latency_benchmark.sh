#!/usr/bin/env bash
# run_vlm_latency_benchmark.sh
#
# VLM latency characterization benchmark.
#
# Sweeps the experiment matrix (conditions A–E) over a fixed image set using
# both the direct/native Edge-LLM invocation path and the ROS
# edge_vlm_ros_node path (where available), keeping model, engine, precision,
# power mode, clocks, image, and prompt text identical between paired runs.
#
# This script does NOT:
#   - Change power mode, clock caps, model size, quantization, or engines.
#   - Fabricate TTFT, decode time, or visual-preprocessing time when the
#     runtime does not expose them; those fields are written as null.
#   - Modify the existing Thor contention benchmark (run_thor_pipeline_benchmarks.sh).
#
# Experiment matrix
# -----------------
#   Condition A : terse identification prompt        , max 16 tokens
#   Condition B : compact structured ODD JSON prompt , max 32 tokens
#   Condition C : compact structured ODD JSON prompt , max 64 tokens
#   Condition D : scene-description prompt           , max 128 tokens
#   Condition E : scene-description prompt           , max 256 tokens
#
# Each condition runs through:
#   - direct  : native llm_inference invocation (requires TENSORRT_EDGE_LLM_ROOT + EDGE_VLM_* env vars)
#   - ros     : vlm_single_shot_client via edge_vlm_cli IPC path (requires edge_vlm_server running)
#
# Fixed image set
# ---------------
# The benchmark uses BENCHMARK_IMAGE_DIR (default: scripts/benchmark/test_fixtures/images).
# Place representative camera frames there alongside the red-panda reference image.
# The red-panda image is downloaded automatically if not present and SKIP_PANDA=false.
#
# Required environment (source scripts/edge_vlm_env.sh before running):
#   TENSORRT_EDGE_LLM_ROOT       root of TensorRT Edge-LLM checkout (binary at build/examples/llm/llm_inference)
#   EDGE_VLM_LLM_ENGINE_DIR      path to LLM engine directory
#   EDGE_VLM_MULTIMODAL_ENGINE_DIR path to multimodal engine directory
#   EDGE_VLM_MODEL_NAME          model identifier string (for metadata)
#   EDGE_VLM_WORKER_SOCKET       IPC socket for edge_vlm_server (default: /tmp/edge_vlm.sock)
#
# Usage
# -----
#   source scripts/edge_vlm_env.sh
#   bash scripts/benchmark/run_vlm_latency_benchmark.sh [OPTIONS]
#
# Options
#   --output-dir DIR         Directory for all output artifacts
#                            (default: /tmp/vlm_latency_bench_TIMESTAMP)
#   --conditions LIST        Comma-separated conditions to run (default: A,B,C,D,E)
#   --paths LIST             Comma-separated paths: direct,ros (default: direct,ros)
#   --warmup N               Warmup iterations per condition/image (default: 1)
#   --iterations N           Measured iterations per condition/image (default: 3)
#   --image-dir DIR          Directory containing fixed benchmark images
#   --skip-panda-download    Skip automatic red-panda reference image download
#   --skip-ros               Alias for --paths direct
#   --skip-direct            Alias for --paths ros
#   --dry-run                Print commands without executing them

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/tmp/vlm_latency_bench_${TIMESTAMP}"
CONDITIONS="A,B,C,D,E"
PATHS="direct,ros"
WARMUP=1
ITERATIONS=3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_DIR="${SCRIPT_DIR}/test_fixtures/images"
SKIP_PANDA_DOWNLOAD=false
DRY_RUN=false

# ── argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)   OUTPUT_DIR="$2";           shift 2 ;;
        --conditions)   CONDITIONS="$2";           shift 2 ;;
        --paths)        PATHS="$2";                shift 2 ;;
        --warmup)       WARMUP="$2";               shift 2 ;;
        --iterations)   ITERATIONS="$2";           shift 2 ;;
        --image-dir)    IMAGE_DIR="$2";            shift 2 ;;
        --skip-panda-download) SKIP_PANDA_DOWNLOAD=true; shift ;;
        --skip-ros)     PATHS="direct";            shift ;;
        --skip-direct)  PATHS="ros";               shift ;;
        --dry-run)      DRY_RUN=true;              shift ;;
        *) echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done

# ── derived paths ─────────────────────────────────────────────────────────────

JSONL_OUT="${OUTPUT_DIR}/vlm_latency_${TIMESTAMP}.jsonl"
REPORT_JSON="${OUTPUT_DIR}/vlm_latency_report.json"
REPORT_TXT="${OUTPUT_DIR}/vlm_latency_report.txt"

# ── experiment matrix ─────────────────────────────────────────────────────────

declare -A CONDITION_PROMPT_ID
declare -A CONDITION_MAX_TOKENS

CONDITION_PROMPT_ID["A"]="terse_id"
CONDITION_PROMPT_ID["B"]="compact_odd_json"
CONDITION_PROMPT_ID["C"]="compact_odd_json"
CONDITION_PROMPT_ID["D"]="scene_description"
CONDITION_PROMPT_ID["E"]="scene_description"

CONDITION_MAX_TOKENS["A"]=16
CONDITION_MAX_TOKENS["B"]=32
CONDITION_MAX_TOKENS["C"]=64
CONDITION_MAX_TOKENS["D"]=128
CONDITION_MAX_TOKENS["E"]=256

# Prompt texts — these must exactly match PROMPT_TEXTS in vlm_latency_report.py
declare -A PROMPT_TEXT
PROMPT_TEXT["terse_id"]="What is in this image?"
PROMPT_TEXT["compact_odd_json"]="Describe this scene as compact JSON with keys: objects, actions, hazards, navigable. Be concise."
PROMPT_TEXT["scene_description"]="You are an autonomous robot perception system. Provide a detailed description of the scene including all visible objects, their positions relative to the robot, any dynamic elements, potential obstacles, and the overall environment type. Be thorough and precise."

# ── utilities ─────────────────────────────────────────────────────────────────

_run() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

_sha256_prefix() {
    # Return first 12 hex chars of SHA-256 of stdin.
    sha256sum | cut -c1-12
}

_now_iso() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

_write_record() {
    # Append one JSON record to JSONL_OUT.
    # Arguments are key=value pairs; values may be null for optional fields.
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] write_record $*"
        return
    fi
    local json="$1"
    echo "${json}" >> "${JSONL_OUT}"
}

_prompt_hash() {
    local prompt_text="$1"
    printf '%s' "${prompt_text}" | sha256sum | cut -c1-12
}

# ── image set setup ───────────────────────────────────────────────────────────

_setup_images() {
    mkdir -p "${IMAGE_DIR}"

    # Red-panda reference image — used in the original smoke test.
    local panda_path="${IMAGE_DIR}/red_panda.jpg"
    if [[ ! -f "${panda_path}" && "${SKIP_PANDA_DOWNLOAD}" == "false" ]]; then
        echo "[setup] Downloading red-panda reference image..."
        if command -v wget &>/dev/null; then
            _run wget -q -O "${panda_path}" \
                "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Red_Panda_%2816862906955%29.jpg/320px-Red_Panda_%2816862906955%29.jpg" \
                || echo "WARNING: panda download failed; skipping" >&2
        elif command -v curl &>/dev/null; then
            _run curl -fsSL -o "${panda_path}" \
                "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Red_Panda_%2816862906955%29.jpg/320px-Red_Panda_%2816862906955%29.jpg" \
                || echo "WARNING: panda download failed; skipping" >&2
        else
            echo "WARNING: wget/curl not available; skipping panda download" >&2
        fi
    fi

    # Enumerate images actually present.
    mapfile -t BENCHMARK_IMAGES < <(
        find "${IMAGE_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort
    )

    if [[ ${#BENCHMARK_IMAGES[@]} -eq 0 ]]; then
        echo "ERROR: no images found in ${IMAGE_DIR}." >&2
        echo "       Add representative camera frames (JPEG/PNG) and optionally" >&2
        echo "       remove --skip-panda-download to fetch the red-panda reference." >&2
        exit 1
    fi

    echo "[setup] Benchmark image set (${#BENCHMARK_IMAGES[@]} images):"
    for img in "${BENCHMARK_IMAGES[@]}"; do
        echo "  ${img}"
    done
}

# ── direct (native llm_inference) invocation ─────────────────────────────────

_run_direct_inference() {
    local condition="$1"
    local prompt_id="$2"
    local max_tokens="$3"
    local prompt_text="$4"
    local image_path="$5"
    local iteration="$6"
    local warmup="$7"
    local run_id="$8"

    local image_id
    image_id="$(basename "${image_path%.*}")"
    local phash
    phash="$(_prompt_hash "${prompt_text}")"
    local recorded_at
    recorded_at="$(_now_iso)"

    # Check that required env vars and the binary are available.
    # Use the same canonical path as run_native_benchmarks.sh:
    #   ${TENSORRT_EDGE_LLM_ROOT}/build/examples/llm/llm_inference
    local llm_inference="${TENSORRT_EDGE_LLM_ROOT:-}/build/examples/llm/llm_inference"
    if [[ -z "${TENSORRT_EDGE_LLM_ROOT:-}" || ! -x "${llm_inference}" ]]; then
        local record
        record=$(python3 -c "
import json, sys
print(json.dumps({
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': sys.argv[1],
    'recorded_at': sys.argv[2],
    'condition': sys.argv[3],
    'path': 'direct',
    'image_id': sys.argv[4],
    'image_path': sys.argv[5],
    'image_width_px': None,
    'image_height_px': None,
    'prompt_id': sys.argv[6],
    'prompt_hash': sys.argv[7],
    'max_output_tokens': int(sys.argv[8]),
    'actual_output_tokens': None,
    'success': False,
    'error': 'TENSORRT_EDGE_LLM_ROOT not set or llm_inference not built — hardware path unavailable',
    'total_latency_ms': None,
    'visual_preprocess_ms': None,
    'ttft_ms': None,
    'decode_ms': None,
    'decode_tokens_per_sec': None,
    'model_name': sys.argv[9] if sys.argv[9] else None,
    'iteration': int(sys.argv[10]),
    'warmup': sys.argv[11] == 'true',
    'tegrastats': None,
}))" \
            "${run_id}" "${recorded_at}" "${condition}" "${image_id}" "${image_path}" \
            "${prompt_id}" "${phash}" "${max_tokens}" \
            "${EDGE_VLM_MODEL_NAME:-}" "${iteration}" "${warmup}")
        _write_record "${record}"
        return
    fi

    # Build a temporary input JSON for llm_inference.
    local input_json
    input_json="$(mktemp /tmp/vlm_bench_input_XXXXXX.json)"
    python3 -c "
import json, sys
obj = {'image': sys.argv[1], 'text': sys.argv[2]}
with open(sys.argv[3], 'w') as f:
    json.dump(obj, f)
" "${image_path}" "${prompt_text}" "${input_json}"

    local output_json
    output_json="$(mktemp /tmp/vlm_bench_output_XXXXXX.json)"

    local t_start t_end
    t_start=$(date +%s%3N)

    local exit_code=0
    _run "${llm_inference}" \
        --engineDir "${EDGE_VLM_LLM_ENGINE_DIR:-}" \
        --multimodalEngineDir "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}" \
        --maxGenerateLength "${max_tokens}" \
        --inputFile "${input_json}" \
        --outputFile "${output_json}" \
        --warmup 0 \
        2>"/tmp/vlm_bench_stderr_${run_id}_${condition}_${iteration}.log" \
        || exit_code=$?

    t_end=$(date +%s%3N)
    local total_latency_ms=$(( t_end - t_start ))

    local error_msg="null"
    local success="true"
    if [[ ${exit_code} -ne 0 ]]; then
        success="false"
        error_msg="\"llm_inference exited with code ${exit_code}\""
        total_latency_ms_json="null"
    else
        total_latency_ms_json="${total_latency_ms}"
    fi

    # Parse output tokens from llm_inference JSON output if available.
    # Stage timings (TTFT, decode time) are extracted when the runtime exposes them;
    # otherwise they remain null — never inferred.
    local record
    record=$(python3 -c "
import json, sys, os

run_id, recorded_at, condition, image_path, image_id = sys.argv[1:6]
prompt_id, phash, max_tokens = sys.argv[6], sys.argv[7], int(sys.argv[8])
total_latency_ms_str = sys.argv[9]
success_str = sys.argv[10]
error_str = sys.argv[11]
model_name_arg = sys.argv[12]
iteration = int(sys.argv[13])
warmup_str = sys.argv[14]
output_json_path = sys.argv[15]

total_latency_ms = float(total_latency_ms_str) if total_latency_ms_str != 'null' else None
success = success_str == 'true'
error = None if error_str == 'null' else error_str.strip('\"')
model_name = model_name_arg if model_name_arg else None
warmup = warmup_str == 'true'

# Parse native output for actual token count and optional stage timings.
actual_output_tokens = None
ttft_ms = None
decode_ms = None
visual_preprocess_ms = None
decode_tokens_per_sec = None

if success and os.path.exists(output_json_path):
    try:
        with open(output_json_path) as f:
            out = json.load(f)
        # Edge-LLM output fields vary by version; extract what is available.
        # Use explicit None checks so zero-valued measurements are not discarded.
        def _first_present(d, *keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None
        actual_output_tokens = _first_present(out, 'outputTokens', 'output_tokens')
        ttft_ms = _first_present(out, 'ttftMs', 'ttft_ms')
        decode_ms = _first_present(out, 'decodeMs', 'decode_ms')
        visual_preprocess_ms = _first_present(out, 'visualMs', 'visual_ms', 'visual_preprocess_ms')
        if actual_output_tokens is not None and decode_ms is not None and decode_ms > 0:
            decode_tokens_per_sec = actual_output_tokens / (decode_ms / 1000.0)
    except Exception:
        pass  # Output file absent or not JSON; all stage timings remain null.

record = {
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': run_id,
    'recorded_at': recorded_at,
    'condition': condition,
    'path': 'direct',
    'image_id': image_id,
    'image_path': image_path,
    'image_width_px': None,
    'image_height_px': None,
    'prompt_id': prompt_id,
    'prompt_hash': phash,
    'max_output_tokens': max_tokens,
    'actual_output_tokens': actual_output_tokens,
    'success': success,
    'error': error,
    'total_latency_ms': total_latency_ms,
    'visual_preprocess_ms': visual_preprocess_ms,
    'ttft_ms': ttft_ms,
    'decode_ms': decode_ms,
    'decode_tokens_per_sec': decode_tokens_per_sec,
    'model_name': model_name,
    'iteration': iteration,
    'warmup': warmup,
    'tegrastats': None,
}
print(json.dumps(record))
" \
        "${run_id}" "${recorded_at}" "${condition}" "${image_path}" "${image_id}" \
        "${prompt_id}" "${phash}" "${max_tokens}" \
        "${total_latency_ms_json:-null}" "${success}" "${error_msg}" \
        "${EDGE_VLM_MODEL_NAME:-}" "${iteration}" "${warmup}" \
        "${output_json}")

    _write_record "${record}"

    rm -f "${input_json}" "${output_json}"
}

# ── ROS path invocation ───────────────────────────────────────────────────────

_run_ros_inference() {
    local condition="$1"
    local prompt_id="$2"
    local max_tokens="$3"
    local prompt_text="$4"
    local image_path="$5"
    local iteration="$6"
    local warmup="$7"
    local run_id="$8"

    local image_id
    image_id="$(basename "${image_path%.*}")"
    local phash
    phash="$(_prompt_hash "${prompt_text}")"
    local recorded_at
    recorded_at="$(_now_iso)"

    # The ROS path requires a running edge_vlm_ros_node and ros2 topic tools.
    # When unavailable, record the attempt as a failure with a descriptive error.
    if ! command -v ros2 &>/dev/null; then
        local record
        record=$(python3 -c "
import json, sys
print(json.dumps({
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': sys.argv[1],
    'recorded_at': sys.argv[2],
    'condition': sys.argv[3],
    'path': 'ros',
    'image_id': sys.argv[4],
    'image_path': sys.argv[5],
    'image_width_px': None,
    'image_height_px': None,
    'prompt_id': sys.argv[6],
    'prompt_hash': sys.argv[7],
    'max_output_tokens': int(sys.argv[8]),
    'actual_output_tokens': None,
    'success': False,
    'error': 'ROS 2 not available in this environment — ros path skipped',
    'total_latency_ms': None,
    'visual_preprocess_ms': None,
    'ttft_ms': None,
    'decode_ms': None,
    'decode_tokens_per_sec': None,
    'model_name': sys.argv[9] if sys.argv[9] else None,
    'iteration': int(sys.argv[10]),
    'warmup': sys.argv[11] == 'true',
    'tegrastats': None,
}))" \
            "${run_id}" "${recorded_at}" "${condition}" "${image_id}" "${image_path}" \
            "${prompt_id}" "${phash}" "${max_tokens}" \
            "${EDGE_VLM_MODEL_NAME:-}" "${iteration}" "${warmup}")
        _write_record "${record}"
        return
    fi

    # Publish the image via the IPC path and capture the result.
    # vlm_single_shot_client wraps edge_vlm_cli with a structured JSON output.
    # Wall-clock timing wraps the full round-trip from invocation to result receipt.
    local result_json
    result_json="$(mktemp /tmp/vlm_bench_ros_result_XXXXXX.json)"

    local t_start t_end
    t_start=$(date +%s%3N)

    local exit_code=0
    _run ros2 run edge_vlm_ros vlm_single_shot_client \
        --socket "${EDGE_VLM_WORKER_SOCKET:-/tmp/edge_vlm.sock}" \
        --image "${image_path}" \
        --prompt "${prompt_text}" \
        --max-tokens "${max_tokens}" \
        --output "${result_json}" \
        --timeout 60 \
        2>"/tmp/vlm_bench_ros_stderr_${run_id}_${condition}_${iteration}.log" \
        || exit_code=$?

    t_end=$(date +%s%3N)
    local total_latency_ms=$(( t_end - t_start ))

    local record
    record=$(python3 -c "
import json, sys, os

run_id, recorded_at, condition, image_path, image_id = sys.argv[1:6]
prompt_id, phash, max_tokens = sys.argv[6], sys.argv[7], int(sys.argv[8])
total_latency_ms = int(sys.argv[9])
exit_code = int(sys.argv[10])
model_name_arg = sys.argv[11]
iteration = int(sys.argv[12])
warmup_str = sys.argv[13]
result_json_path = sys.argv[14]

success = exit_code == 0
error = None if success else f'ros client exited with code {exit_code}'
model_name = model_name_arg if model_name_arg else None
warmup = warmup_str == 'true'
total_ms = float(total_latency_ms) if success else None

# ROS path does not expose stage timings (TTFT, decode, visual) — all null.
actual_output_tokens = None
if success and os.path.exists(result_json_path):
    try:
        with open(result_json_path) as f:
            out = json.load(f)
        actual_output_tokens = out.get('output_tokens') or out.get('actualOutputTokens')
    except Exception:
        pass

record = {
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': run_id,
    'recorded_at': recorded_at,
    'condition': condition,
    'path': 'ros',
    'image_id': image_id,
    'image_path': image_path,
    'image_width_px': None,
    'image_height_px': None,
    'prompt_id': prompt_id,
    'prompt_hash': phash,
    'max_output_tokens': max_tokens,
    'actual_output_tokens': actual_output_tokens,
    'success': success,
    'error': error,
    'total_latency_ms': total_ms,
    'visual_preprocess_ms': None,
    'ttft_ms': None,
    'decode_ms': None,
    'decode_tokens_per_sec': None,
    'model_name': model_name,
    'iteration': iteration,
    'warmup': warmup,
    'tegrastats': None,
}
print(json.dumps(record))
" \
        "${run_id}" "${recorded_at}" "${condition}" "${image_path}" "${image_id}" \
        "${prompt_id}" "${phash}" "${max_tokens}" \
        "${total_latency_ms}" "${exit_code}" \
        "${EDGE_VLM_MODEL_NAME:-}" "${iteration}" "${warmup}" \
        "${result_json}")

    _write_record "${record}"
    rm -f "${result_json}"
}

# ── main ──────────────────────────────────────────────────────────────────────

main() {
    echo "[vlm-latency-bench] Run ID: ${TIMESTAMP}"
    echo "[vlm-latency-bench] Output: ${OUTPUT_DIR}"

    if [[ "${DRY_RUN}" != "true" ]]; then
        mkdir -p "${OUTPUT_DIR}"
    fi

    # Resolve image set.
    BENCHMARK_IMAGES=()
    _setup_images

    # Parse condition and path lists.
    IFS=',' read -ra CONDITION_LIST <<< "${CONDITIONS}"
    IFS=',' read -ra PATH_LIST <<< "${PATHS}"

    local run_id="${TIMESTAMP}"

    echo "[vlm-latency-bench] Conditions: ${CONDITIONS}"
    echo "[vlm-latency-bench] Paths: ${PATHS}"
    echo "[vlm-latency-bench] Warmup: ${WARMUP}, Iterations: ${ITERATIONS}"
    echo "[vlm-latency-bench] Images: ${#BENCHMARK_IMAGES[@]}"

    for condition in "${CONDITION_LIST[@]}"; do
        local prompt_id="${CONDITION_PROMPT_ID[${condition}]}"
        local max_tokens="${CONDITION_MAX_TOKENS[${condition}]}"
        local prompt_text="${PROMPT_TEXT[${prompt_id}]}"

        echo "[vlm-latency-bench] Condition ${condition}: prompt=${prompt_id}, max_tokens=${max_tokens}"

        for path in "${PATH_LIST[@]}"; do
            for image_path in "${BENCHMARK_IMAGES[@]}"; do
                echo "[vlm-latency-bench]   ${path} / $(basename "${image_path}")"

                # Warmup iterations
                for (( i=0; i<WARMUP; i++ )); do
                    if [[ "${path}" == "direct" ]]; then
                        _run_direct_inference \
                            "${condition}" "${prompt_id}" "${max_tokens}" \
                            "${prompt_text}" "${image_path}" "${i}" "true" "${run_id}"
                    elif [[ "${path}" == "ros" ]]; then
                        _run_ros_inference \
                            "${condition}" "${prompt_id}" "${max_tokens}" \
                            "${prompt_text}" "${image_path}" "${i}" "true" "${run_id}"
                    fi
                done

                # Measured iterations
                for (( i=0; i<ITERATIONS; i++ )); do
                    if [[ "${path}" == "direct" ]]; then
                        _run_direct_inference \
                            "${condition}" "${prompt_id}" "${max_tokens}" \
                            "${prompt_text}" "${image_path}" "${i}" "false" "${run_id}"
                    elif [[ "${path}" == "ros" ]]; then
                        _run_ros_inference \
                            "${condition}" "${prompt_id}" "${max_tokens}" \
                            "${prompt_text}" "${image_path}" "${i}" "false" "${run_id}"
                    fi
                done
            done
        done
    done

    echo "[vlm-latency-bench] Generating report..."
    _run python3 "${SCRIPT_DIR}/vlm_latency_report.py" \
        --input "${JSONL_OUT}" \
        --output "${REPORT_JSON}" \
        --text "${REPORT_TXT}"

    echo "[vlm-latency-bench] Done."
    echo "[vlm-latency-bench] Raw records: ${JSONL_OUT}"
    echo "[vlm-latency-bench] JSON report: ${REPORT_JSON}"
    echo "[vlm-latency-bench] Text report: ${REPORT_TXT}"
}

main "$@"
