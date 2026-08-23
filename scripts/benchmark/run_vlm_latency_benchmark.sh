#!/usr/bin/env bash
# run_vlm_latency_benchmark.sh
#
# VLM latency characterization benchmark.
#
# Sweeps the experiment matrix (conditions A–E) over a fixed image set using
# both the direct/native Edge-LLM invocation path and the IPC (standalone
# edge_vlm_server) path, keeping model, engine, precision, power mode, clocks,
# image, and prompt text identical between paired runs.
#
# NOTE on lifecycle semantics
# ---------------------------
#   direct : launches a fresh llm_inference process per inference — includes
#            cold-start (process init, engine load, tokenizer init). This is
#            per-process cold-start latency.
#   ipc    : sends requests to an already-running, warmed edge_vlm_server via
#            IPC socket (vlm_single_shot_client → edge_vlm_cli). This is
#            steady-state (persistent server) latency.
#   These two paths are NOT directly comparable as "overhead" because they
#   measure different lifecycle phases. The report labels them accordingly.
#   Both paths are recorded so each can be analysed independently.
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
#   - ipc     : vlm_single_shot_client → edge_vlm_cli → running edge_vlm_server IPC socket
#
# Fixed image set
# ---------------
# The benchmark uses BENCHMARK_IMAGE_DIR (default: scripts/benchmark/test_fixtures/images).
# Images must have neutral names (image_001.jpg, image_002.jpg …).  On the first
# run, the script tries to copy known-good fixtures from the pinned
# TensorRT Edge-LLM checkout — probing examples/multimodal/pics/ first (the
# verified Thor layout), then examples/vlm/data/images/ as a fallback —
# before falling back to a one-time download of the NVIDIA red-panda reference
# saved as image_001.jpg.
# Every image is validated (non-zero size, JPEG/PNG magic bytes) before use.
# Invalid images cause a clear error; they are never silently admitted.
# A SHA-256 content hash is recorded with each inference for reproducibility.
#
# Native llm_inference profiling
# --------------------------------
# For direct-path runs, llm_inference is invoked with --dumpProfile
# --profileOutputFile so the NVIDIA-authoritative profile JSON is preserved as a
# raw artifact alongside the response JSON.  Shell wall time (cold_start_total_ms)
# is recorded separately from NVIDIA-emitted stage metrics.
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
#   --paths LIST             Comma-separated paths: direct,ipc (default: direct,ipc)
#   --warmup N               Warmup iterations per condition/image (default: 1)
#   --iterations N           Measured iterations per condition/image (default: 3)
#   --image-dir DIR          Directory containing fixed benchmark images
#   --skip-image-download    Skip automatic reference image download/copy
#   --skip-panda-download    Alias for --skip-image-download (backward compat)
#   --skip-ipc               Alias for --paths direct
#   --skip-direct            Alias for --paths ipc
#   --dry-run                Print commands without executing them

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/tmp/vlm_latency_bench_${TIMESTAMP}"
CONDITIONS="A,B,C,D,E"
PATHS="direct,ipc"
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
        --skip-image-download) SKIP_PANDA_DOWNLOAD=true; shift ;;
        --skip-panda-download) SKIP_PANDA_DOWNLOAD=true; shift ;;  # backward compat alias
        --skip-ipc)     PATHS="direct";            shift ;;
        --skip-direct)  PATHS="ipc";               shift ;;
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

_image_content_hash() {
    # Return first 12 hex chars of SHA-256 of the given image file.
    sha256sum "$1" | cut -c1-12
}

_validate_image() {
    # Validate that a file is a non-zero-size JPEG or PNG.
    # Prints a human-readable error and returns 1 on failure.
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: image not found: ${path}" >&2; return 1
    fi
    local size
    size=$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}" 2>/dev/null || echo 0)
    if [[ "${size}" -eq 0 ]]; then
        echo "ERROR: image is zero bytes (download may have failed): ${path}" >&2; return 1
    fi
    # Check magic bytes: JPEG = FF D8 FF (3 bytes), PNG = 89 50 4E 47 (4 bytes).
    # Read 4 bytes so the full PNG signature is captured.
    local magic
    magic=$(xxd -p -l 4 "${path}" 2>/dev/null || od -A n -N 4 -t x1 "${path}" 2>/dev/null | tr -d ' \n')
    case "${magic,,}" in
        ffd8ff*)   ;;  # JPEG (starts with FF D8 FF)
        89504e47*) ;;  # PNG  (starts with 89 50 4E 47)
        *)
            echo "ERROR: ${path}: not a valid JPEG or PNG (magic=${magic}); " \
                 "file may be an HTML error page or truncated download" >&2
            return 1
            ;;
    esac
    return 0
}

# ── image set setup ───────────────────────────────────────────────────────────

_setup_images() {
    mkdir -p "${IMAGE_DIR}"

    if [[ "${SKIP_PANDA_DOWNLOAD}" == "false" ]]; then
        # Prefer copying known-good fixtures from the pinned TensorRT Edge-LLM checkout
        # (neutral file names image_001.jpg, image_002.jpg, …).
        local copied=0
        # Probe known fixture paths in order of preference:
        #   1. examples/multimodal/pics/  — path verified on the tested Thor checkout
        #   2. examples/vlm/data/images/  — alternate layout seen in some releases
        local edgellm_img_dir=""
        for _candidate in \
            "${TENSORRT_EDGE_LLM_ROOT:-}/examples/multimodal/pics" \
            "${TENSORRT_EDGE_LLM_ROOT:-}/examples/vlm/data/images"
        do
            if [[ -d "${_candidate}" ]]; then
                edgellm_img_dir="${_candidate}"
                break
            fi
        done
        if [[ -n "${edgellm_img_dir}" ]]; then
            local idx=1
            while IFS= read -r -d '' src; do
                local dst
                dst="${IMAGE_DIR}/$(printf 'image_%03d' "${idx}").${src##*.}"
                if [[ ! -f "${dst}" ]]; then
                    cp "${src}" "${dst}"
                    echo "[setup] Copied $(basename "${src}") → $(basename "${dst}")"
                fi
                (( idx++ ))
                (( copied++ ))
            done < <(find "${edgellm_img_dir}" -maxdepth 1 -type f \
                          \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
                          -print0 | sort -z)
        fi

        # Fall back: download NVIDIA red-panda reference as image_001.jpg.
        local ref_path="${IMAGE_DIR}/image_001.jpg"
        if [[ ${copied} -eq 0 && ! -f "${ref_path}" ]]; then
            echo "[setup] Downloading reference image as image_001.jpg ..."
            if command -v wget &>/dev/null; then
                _run wget -q -O "${ref_path}" \
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Red_Panda_%2816862906955%29.jpg/320px-Red_Panda_%2816862906955%29.jpg" \
                    || echo "WARNING: reference image download failed; skipping" >&2
            elif command -v curl &>/dev/null; then
                _run curl -fsSL -o "${ref_path}" \
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Red_Panda_%2816862906955%29.jpg/320px-Red_Panda_%2816862906955%29.jpg" \
                    || echo "WARNING: reference image download failed; skipping" >&2
            else
                echo "WARNING: wget/curl not available; skipping reference image download" >&2
            fi
        fi
    fi

    # Enumerate candidate images.
    mapfile -t _CANDIDATE_IMAGES < <(
        find "${IMAGE_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort
    )

    # Validate each candidate; exclude invalid ones with a clear error.
    BENCHMARK_IMAGES=()
    for img in "${_CANDIDATE_IMAGES[@]}"; do
        if _validate_image "${img}"; then
            BENCHMARK_IMAGES+=("${img}")
        else
            echo "WARNING: excluding invalid image from benchmark: ${img}" >&2
        fi
    done

    if [[ ${#BENCHMARK_IMAGES[@]} -eq 0 ]]; then
        echo "ERROR: no valid images found in ${IMAGE_DIR}." >&2
        echo "       Add JPEG/PNG camera frames with neutral names (image_001.jpg, …)" >&2
        echo "       or remove --skip-image-download to fetch the reference fixture." >&2
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
    local content_hash
    content_hash="$(_image_content_hash "${image_path}" 2>/dev/null || echo "null")"

    # Check that required env vars and the binary are available.
    # Use the same canonical path as run_native_benchmarks.sh:
    #   ${TENSORRT_EDGE_LLM_ROOT}/build/examples/llm/llm_inference
    local llm_inference="${TENSORRT_EDGE_LLM_ROOT:-}/build/examples/llm/llm_inference"
    if [[ -z "${TENSORRT_EDGE_LLM_ROOT:-}" || ! -x "${llm_inference}" ]]; then
        local record
        record=$(python3 -c "
import json, sys
ch = sys.argv[12]
print(json.dumps({
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': sys.argv[1],
    'recorded_at': sys.argv[2],
    'condition': sys.argv[3],
    'path': 'direct',
    'lifecycle_semantics': 'cold_start',
    'image_id': sys.argv[4],
    'image_path': sys.argv[5],
    'image_width_px': None,
    'image_height_px': None,
    'content_hash': ch if ch != 'null' else None,
    'prompt_id': sys.argv[6],
    'prompt_hash': sys.argv[7],
    'max_output_tokens': int(sys.argv[8]),
    'actual_output_tokens': None,
    'finish_reason': None,
    'output_text': None,
    'output_words': None,
    'inference_seconds': None,
    'success': False,
    'error': 'TENSORRT_EDGE_LLM_ROOT not set or llm_inference not built — hardware path unavailable',
    'cold_start_total_ms': None,
    'total_latency_ms': None,
    'vision_encoder_ms': None,
    'prefill_ms': None,
    'decode_ms': None,
    'decode_tokens_per_sec': None,
    'average_time_per_token_ms': None,
    'llm_generation_total_gpu_time_ms': None,
    'native_response_path': None,
    'native_profile_path': None,
    'model_name': sys.argv[9] if sys.argv[9] else None,
    'iteration': int(sys.argv[10]),
    'warmup': sys.argv[11] == 'true',
    'tegrastats': None,
}))" \
            "${run_id}" "${recorded_at}" "${condition}" "${image_id}" "${image_path}" \
            "${prompt_id}" "${phash}" "${max_tokens}" \
            "${EDGE_VLM_MODEL_NAME:-}" "${iteration}" "${warmup}" "${content_hash}")
        _write_record "${record}"
        return
    fi

    # Build a temporary input JSON for llm_inference using the pinned NVIDIA VLM
    # request schema: requests -> messages -> content[{type,image},{type,text}].
    # This is the same shape used by tests/test_cases/vlm_basic.json in the
    # TensorRT Edge-LLM checkout.
    local input_json
    input_json="$(mktemp /tmp/vlm_bench_input_XXXXXX.json)"
    python3 -c "
import json, sys
obj = {
    'requests': [
        {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': sys.argv[1]},
                        {'type': 'text',  'text':  sys.argv[2]},
                    ]
                }
            ]
        }
    ]
}
with open(sys.argv[3], 'w') as f:
    json.dump(obj, f)
" "${image_path}" "${prompt_text}" "${input_json}"

    # Named artifacts for this inference (preserved for post-analysis).
    # Include image_id and warmup/measured phase to avoid collisions across images
    # and between warmup and measured iterations.
    local phase="measured"
    [[ "${warmup}" == "true" ]] && phase="warmup"
    local artifact_base="${OUTPUT_DIR}/direct_${condition}_${image_id}_${phase}_iter${iteration}"
    local output_json="${artifact_base}_response.json"
    local profile_json="${artifact_base}_profile.json"

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
        --dumpProfile \
        --profileOutputFile "${profile_json}" \
        2>"/tmp/vlm_bench_stderr_${run_id}_${condition}_${iteration}.log" \
        || exit_code=$?

    t_end=$(date +%s%3N)
    local cold_start_total_ms=$(( t_end - t_start ))

    local error_msg="null"
    local success="true"
    local cold_start_total_ms_json="${cold_start_total_ms}"
    if [[ ${exit_code} -ne 0 ]]; then
        success="false"
        error_msg="\"llm_inference exited with code ${exit_code}\""
        cold_start_total_ms_json="null"
    fi

    # Parse output tokens, finish_reason, and NVIDIA-authoritative stage timings
    # from llm_inference JSON artifacts.  Only fields actually present in the
    # runtime output are extracted; unknown/unavailable fields remain null.
    # Shell wall time (cold_start_total_ms) is kept separate from model/runtime
    # inference metrics.
    local record
    record=$(python3 -c "
import json, sys, os

run_id, recorded_at, condition, image_path, image_id = sys.argv[1:6]
prompt_id, phash, max_tokens = sys.argv[6], sys.argv[7], int(sys.argv[8])
cold_start_total_ms_str = sys.argv[9]
success_str = sys.argv[10]
error_str = sys.argv[11]
model_name_arg = sys.argv[12]
iteration = int(sys.argv[13])
warmup_str = sys.argv[14]
output_json_path = sys.argv[15]
profile_json_path = sys.argv[16]
content_hash_arg = sys.argv[17]

cold_start_total_ms = float(cold_start_total_ms_str) if cold_start_total_ms_str != 'null' else None
success = success_str == 'true'
error = None if error_str == 'null' else error_str.strip('\"')
model_name = model_name_arg if model_name_arg else None
warmup = warmup_str == 'true'
content_hash = content_hash_arg if content_hash_arg != 'null' else None

# Records existence of preserved artifacts for reference.
response_path = output_json_path if os.path.exists(output_json_path) else None
profile_path  = profile_json_path if os.path.exists(profile_json_path) else None

# Parse NVIDIA-authoritative fields from native response and profile artifacts.
# Only fields actually present are extracted; nothing is inferred or fabricated.
actual_output_tokens = None
finish_reason = None
prefill_ms = None
decode_ms = None
vision_encoder_ms = None
decode_tokens_per_sec = None
average_time_per_token_ms = None
llm_generation_total_gpu_time_ms = None
output_text = None

def _first_present(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

if success and response_path:
    try:
        with open(response_path) as f:
            out = json.load(f)
        # Thor's llm_inference response is {"responses": [{...}]}.
        # Unwrap the first responses[] entry; also accept a flat top-level dict
        # as a fallback for alternative runtime shapes.
        entry = out
        responses_list = out.get('responses') if isinstance(out, dict) else None
        if isinstance(responses_list, list) and responses_list:
            entry = responses_list[0]
        # Scalar response fields
        actual_output_tokens = _first_present(entry, 'outputTokens', 'output_tokens', 'numOutputTokens')
        finish_reason = _first_present(entry, 'finishReason', 'finish_reason')
        output_text = _first_present(entry, 'outputText', 'output_text', 'text', 'response')
    except Exception:
        pass

# Prefer NVIDIA-emitted profile data for stage timings (authoritative).
# Parse the pinned TensorRT Edge-LLM nested profile schema:
#   generation.generated_tokens   -> actual_output_tokens (if not already set)
#   generation.tokens_per_second  -> decode_tokens_per_sec
#   generation.average_time_per_token_ms -> average_time_per_token_ms
#   generation.total_time_ms      -> decode_ms
#   prefill.average_time_per_run_ms -> prefill_ms
#   stages[stage_id='vision_encoder'].average_time_per_run_ms -> vision_encoder_ms
#   stages[stage_id='llm_generation'].total_gpu_time_ms -> llm_generation_total_gpu_time_ms
if success and profile_path:
    try:
        with open(profile_path) as f:
            prof = json.load(f)
        gen = prof.get('generation') if isinstance(prof, dict) else None
        if isinstance(gen, dict):
            if actual_output_tokens is None:
                actual_output_tokens = gen.get('generated_tokens')
            v = gen.get('tokens_per_second')
            if v is not None:
                decode_tokens_per_sec = v
            average_time_per_token_ms = gen.get('average_time_per_token_ms')
            v = gen.get('total_time_ms')
            if v is not None:
                decode_ms = v
        prefill = prof.get('prefill') if isinstance(prof, dict) else None
        if isinstance(prefill, dict):
            v = prefill.get('average_time_per_run_ms')
            if v is not None:
                prefill_ms = v
        stages = prof.get('stages') if isinstance(prof, dict) else None
        if isinstance(stages, list):
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                sid = stage.get('stage_id')
                if sid == 'vision_encoder':
                    v = stage.get('average_time_per_run_ms')
                    if v is not None:
                        vision_encoder_ms = v
                elif sid == 'llm_generation':
                    v = stage.get('total_gpu_time_ms')
                    if v is not None:
                        llm_generation_total_gpu_time_ms = v
    except Exception:
        pass

record = {
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': run_id,
    'recorded_at': recorded_at,
    'condition': condition,
    'path': 'direct',
    'lifecycle_semantics': 'cold_start',
    'image_id': image_id,
    'image_path': image_path,
    'image_width_px': None,
    'image_height_px': None,
    'content_hash': content_hash,
    'prompt_id': prompt_id,
    'prompt_hash': phash,
    'max_output_tokens': max_tokens,
    'actual_output_tokens': actual_output_tokens,
    'finish_reason': finish_reason,
    'output_text': output_text,
    'output_words': None,
    'inference_seconds': None,
    'success': success,
    'error': error,
    'cold_start_total_ms': cold_start_total_ms,
    'total_latency_ms': None,
    'vision_encoder_ms': vision_encoder_ms,
    'prefill_ms': prefill_ms,
    'decode_ms': decode_ms,
    'decode_tokens_per_sec': decode_tokens_per_sec,
    'average_time_per_token_ms': average_time_per_token_ms,
    'llm_generation_total_gpu_time_ms': llm_generation_total_gpu_time_ms,
    'native_response_path': response_path,
    'native_profile_path': profile_path,
    'model_name': model_name,
    'iteration': iteration,
    'warmup': warmup,
    'tegrastats': None,
}
print(json.dumps(record))
" \
        "${run_id}" "${recorded_at}" "${condition}" "${image_path}" "${image_id}" \
        "${prompt_id}" "${phash}" "${max_tokens}" \
        "${cold_start_total_ms_json:-null}" "${success}" "${error_msg}" \
        "${EDGE_VLM_MODEL_NAME:-}" "${iteration}" "${warmup}" \
        "${output_json}" "${profile_json}" "${content_hash}")

    _write_record "${record}"

    rm -f "${input_json}"
    # output_json and profile_json are preserved as named artifacts.
}

# ── IPC path invocation (persistent edge_vlm_server) ─────────────────────────

_run_ipc_inference() {
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
    local content_hash
    content_hash="$(_image_content_hash "${image_path}" 2>/dev/null || echo "null")"

    # The ipc path invokes vlm_single_shot_client via `ros2 run` because the
    # script is installed into the ROS lib directory (lib/edge_vlm_ros/).
    # `ros2` is required only to locate and launch the installed script;
    # vlm_single_shot_client itself does not depend on ROS at runtime — it
    # communicates with edge_vlm_server directly via edge_vlm_cli.
    # A running edge_vlm_server on EDGE_VLM_WORKER_SOCKET is also required.
    if ! command -v ros2 &>/dev/null; then
        local record
        record=$(python3 -c "
import json, sys
ch = sys.argv[12]
print(json.dumps({
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': sys.argv[1],
    'recorded_at': sys.argv[2],
    'condition': sys.argv[3],
    'path': 'ipc',
    'lifecycle_semantics': 'persistent',
    'image_id': sys.argv[4],
    'image_path': sys.argv[5],
    'image_width_px': None,
    'image_height_px': None,
    'content_hash': ch if ch != 'null' else None,
    'prompt_id': sys.argv[6],
    'prompt_hash': sys.argv[7],
    'max_output_tokens': int(sys.argv[8]),
    'actual_output_tokens': None,
    'finish_reason': None,
    'output_text': None,
    'output_words': None,
    'inference_seconds': None,
    'success': False,
    'error': 'ros2 not available — ipc path (vlm_single_shot_client) skipped',
    'cold_start_total_ms': None,
    'total_latency_ms': None,
    'vision_encoder_ms': None,
    'prefill_ms': None,
    'decode_ms': None,
    'decode_tokens_per_sec': None,
    'average_time_per_token_ms': None,
    'llm_generation_total_gpu_time_ms': None,
    'native_response_path': None,
    'native_profile_path': None,
    'model_name': sys.argv[9] if sys.argv[9] else None,
    'iteration': int(sys.argv[10]),
    'warmup': sys.argv[11] == 'true',
    'tegrastats': None,
}))" \
            "${run_id}" "${recorded_at}" "${condition}" "${image_id}" "${image_path}" \
            "${prompt_id}" "${phash}" "${max_tokens}" \
            "${EDGE_VLM_MODEL_NAME:-}" "${iteration}" "${warmup}" "${content_hash}")
        _write_record "${record}"
        return
    fi

    # Send the request to the already-running edge_vlm_server via IPC and capture
    # the result.  vlm_single_shot_client wraps edge_vlm_cli with structured JSON
    # output.  Wall-clock timing wraps the full client round-trip.
    # The result artifact is preserved in OUTPUT_DIR with a collision-safe name.
    local phase="measured"
    [[ "${warmup}" == "true" ]] && phase="warmup"
    local result_json="${OUTPUT_DIR}/ipc_${condition}_${image_id}_${phase}_iter${iteration}_result.json"

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
        2>"/tmp/vlm_bench_ipc_stderr_${run_id}_${condition}_${iteration}.log" \
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
content_hash_arg = sys.argv[15]

success = exit_code == 0
error = None if success else f'ipc client exited with code {exit_code}'
model_name = model_name_arg if model_name_arg else None
warmup = warmup_str == 'true'
total_ms = float(total_latency_ms) if success else None
content_hash = content_hash_arg if content_hash_arg != 'null' else None

# IPC path does not expose stage timings (TTFT, decode, visual) — all null.
# Preserve output_text, output_words, and inference_seconds from the client
# artifact so output semantics are available for analysis.
# actual_output_tokens is null when the backend does not report a real token count.
actual_output_tokens = None
output_text = None
output_words = None
inference_seconds = None
if success and os.path.exists(result_json_path):
    try:
        with open(result_json_path) as f:
            out = json.load(f)
        actual_output_tokens = out.get('output_tokens') or out.get('actualOutputTokens')
        output_text = out.get('output_text')
        output_words = out.get('output_words')
        raw_secs = out.get('inference_seconds')
        inference_seconds = float(raw_secs) if raw_secs is not None else None
    except Exception:
        pass

record = {
    'schema_version': '1',
    'record_type': 'inference',
    'run_id': run_id,
    'recorded_at': recorded_at,
    'condition': condition,
    'path': 'ipc',
    'lifecycle_semantics': 'persistent',
    'image_id': image_id,
    'image_path': image_path,
    'image_width_px': None,
    'image_height_px': None,
    'content_hash': content_hash,
    'prompt_id': prompt_id,
    'prompt_hash': phash,
    'max_output_tokens': max_tokens,
    'actual_output_tokens': actual_output_tokens,
    'finish_reason': None,
    'output_text': output_text,
    'output_words': output_words,
    'inference_seconds': inference_seconds,
    'success': success,
    'error': error,
    'cold_start_total_ms': None,
    'total_latency_ms': total_ms,
    'vision_encoder_ms': None,
    'prefill_ms': None,
    'decode_ms': None,
    'decode_tokens_per_sec': None,
    'average_time_per_token_ms': None,
    'llm_generation_total_gpu_time_ms': None,
    'native_response_path': None,
    'native_profile_path': None,
    'ipc_result_path': result_json_path if success and os.path.exists(result_json_path) else None,
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
        "${result_json}" "${content_hash}")

    _write_record "${record}"
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
                    elif [[ "${path}" == "ipc" ]]; then
                        _run_ipc_inference \
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
                    elif [[ "${path}" == "ipc" ]]; then
                        _run_ipc_inference \
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
