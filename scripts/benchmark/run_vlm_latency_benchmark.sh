#!/usr/bin/env bash
# VLM single-frame latency characterization benchmark.
#
# Conditions A-E vary prompt style and output-token cap while holding the input
# image set and runtime configuration fixed. The optional direct path measures
# per-process cold-start behavior; the IPC path measures a persistent server.
set -euo pipefail

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="/tmp/vlm_latency_bench_${TIMESTAMP}"
IMAGE_DIR="${SCRIPT_DIR}/test_fixtures/images"
CONDITIONS="A,B,C,D,E"
PATHS="direct,ipc"
WARMUP=1
ITERATIONS=3
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --image-dir) IMAGE_DIR="$2"; shift 2 ;;
        --conditions) CONDITIONS="$2"; shift 2 ;;
        --paths) PATHS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --skip-ipc) PATHS="direct"; shift ;;
        --skip-direct) PATHS="ipc"; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-image-download|--skip-panda-download) shift ;; # retained compatibility; fixtures are no longer downloaded here
        *) echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
JSONL_OUT="${OUTPUT_DIR}/vlm_latency_${TIMESTAMP}.jsonl"
REPORT_JSON="${OUTPUT_DIR}/vlm_latency_report.json"
REPORT_TXT="${OUTPUT_DIR}/vlm_latency_report.txt"

# Prompt IDs intentionally describe generic output style rather than an application.
declare -A CONDITION_PROMPT_ID=(
    [A]="terse_id"
    [B]="compact_scene_json"
    [C]="compact_scene_json"
    [D]="scene_description"
    [E]="scene_description"
)
declare -A CONDITION_MAX_TOKENS=([A]=16 [B]=32 [C]=64 [D]=128 [E]=256)
declare -A PROMPT_TEXT=(
    [terse_id]="What is in this image?"
    [compact_scene_json]="Describe this scene as compact JSON with keys: objects, actions, hazards, navigable. Be concise."
    [scene_description]="Provide a detailed description of the scene including visible objects, their relative positions, dynamic elements, potential obstacles, and the overall environment type. Be thorough and precise."
)

now_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
prompt_hash() { printf '%s' "$1" | sha256sum | cut -c1-12; }
content_hash() { sha256sum "$1" | awk '{print substr($1,1,12)}'; }

mapfile -t IMAGES < <(find "${IMAGE_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort)
if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "ERROR: no benchmark JPEG/PNG images found in ${IMAGE_DIR}" >&2
    exit 1
fi

write_record() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] record: $1"
    else
        printf '%s\n' "$1" >> "${JSONL_OUT}"
    fi
}

build_direct_input() {
    local image="$1" prompt="$2" output="$3"
    python3 - "$image" "$prompt" "$output" <<'PY'
import json, sys
image, prompt, output = sys.argv[1:]
obj = {"requests": [{"messages": [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": prompt},
]}]}]}
with open(output, "w", encoding="utf-8") as fh:
    json.dump(obj, fh)
PY
}

emit_record() {
    local condition="$1" path="$2" image="$3" prompt_id="$4" max_tokens="$5"
    local iteration="$6" warmup="$7" artifact="$8" cold_ms="$9" total_ms="${10}" profile="${11}" success="${12}" error="${13}"
    python3 - "$condition" "$path" "$image" "$prompt_id" "$max_tokens" "$iteration" "$warmup" "$artifact" "$cold_ms" "$total_ms" "$profile" "$success" "$error" "$(now_iso)" "$(prompt_hash "${PROMPT_TEXT[$prompt_id]}")" "$(content_hash "$image")" <<'PY'
import json, os, sys
(condition, path, image, prompt_id, max_tokens, iteration, warmup, artifact,
 cold_ms, total_ms, profile_path, success, error, recorded_at, prompt_hash, content_hash) = sys.argv[1:]

def num(v):
    return None if v in ("", "null") else float(v)

record = {
    "schema_version": "1",
    "record_type": "inference",
    "run_id": os.environ.get("BENCH_RUN_ID", "vlm-latency"),
    "recorded_at": recorded_at,
    "condition": condition,
    "path": path,
    "lifecycle_semantics": "cold_start" if path == "direct" else "persistent",
    "image_id": os.path.splitext(os.path.basename(image))[0],
    "image_path": image,
    "image_width_px": None,
    "image_height_px": None,
    "content_hash": content_hash,
    "prompt_id": prompt_id,
    "prompt_hash": prompt_hash,
    "max_output_tokens": int(max_tokens),
    "actual_output_tokens": None,
    "finish_reason": None,
    "output_text": None,
    "output_words": None,
    "inference_seconds": None,
    "success": success == "true",
    "error": error or None,
    "cold_start_total_ms": num(cold_ms),
    "total_latency_ms": num(total_ms),
    "vision_encoder_ms": None,
    "prefill_ms": None,
    "ttft_ms": None,
    "decode_ms": None,
    "decode_tokens_per_sec": None,
    "average_time_per_token_ms": None,
    "llm_generation_total_gpu_time_ms": None,
    "native_response_path": artifact if path == "direct" and artifact else None,
    "native_profile_path": profile_path or None,
    "ipc_result_path": artifact if path == "ipc" and artifact else None,
    "model_name": os.environ.get("EDGE_VLM_MODEL_NAME", "unknown"),
    "iteration": int(iteration),
    "warmup": warmup == "true",
    "tegrastats": None,
}

if artifact and os.path.isfile(artifact):
    try:
        data = json.load(open(artifact, encoding="utf-8"))
        record["success"] = bool(data.get("success", record["success"]))
        record["output_text"] = data.get("output_text") or data.get("text")
        record["actual_output_tokens"] = data.get("output_tokens") or data.get("generated_tokens")
        record["output_words"] = data.get("output_words")
        record["inference_seconds"] = data.get("inference_seconds")
        record["finish_reason"] = data.get("finish_reason")
        if data.get("error"):
            record["error"] = data["error"]
    except Exception:
        pass

if profile_path and os.path.isfile(profile_path):
    try:
        p = json.load(open(profile_path, encoding="utf-8"))
        generation = p.get("generation", {})
        prefill = p.get("prefill", {})
        record["actual_output_tokens"] = generation.get("generated_tokens", record["actual_output_tokens"])
        record["decode_tokens_per_sec"] = generation.get("tokens_per_second")
        record["decode_ms"] = generation.get("total_time_ms")
        record["average_time_per_token_ms"] = generation.get("average_time_per_token_ms")
        record["prefill_ms"] = prefill.get("average_time_per_run_ms")
        for stage in p.get("stages", []):
            sid = stage.get("stage_id") or stage.get("name")
            if sid == "vision_encoder":
                record["vision_encoder_ms"] = stage.get("average_time_per_run_ms")
            elif sid == "llm_generation":
                record["llm_generation_total_gpu_time_ms"] = stage.get("total_gpu_time_ms")
    except Exception:
        pass
print(json.dumps(record))
PY
}

run_direct() {
    local condition="$1" image="$2" prompt_id="$3" max_tokens="$4" iteration="$5" warmup="$6"
    local binary="${TENSORRT_EDGE_LLM_ROOT:-}/build/examples/llm/llm_inference"
    local dir="${OUTPUT_DIR}/${condition}/direct/$(basename "${image%.*}")"
    mkdir -p "$dir"
    local input="$dir/input_${iteration}_${warmup}.json" output="$dir/output_${iteration}_${warmup}.json" profile="$dir/profile_${iteration}_${warmup}.json"
    build_direct_input "$image" "${PROMPT_TEXT[$prompt_id]}" "$input"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] $binary --engineDir ... --inputFile $input"
        return
    fi
    if [[ ! -x "$binary" ]]; then
        write_record "$(emit_record "$condition" direct "$image" "$prompt_id" "$max_tokens" "$iteration" "$warmup" "" null null "" false "llm_inference not found")"
        return
    fi

    local start end rc=0
    start=$(date +%s%N)
    "$binary" \
        --engineDir "${EDGE_VLM_LLM_ENGINE_DIR:?EDGE_VLM_LLM_ENGINE_DIR is required}" \
        --multimodalEngineDir "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:?EDGE_VLM_MULTIMODAL_ENGINE_DIR is required}" \
        --maxGenerateLength "$max_tokens" \
        --inputFile "$input" \
        --outputFile "$output" \
        --warmup 0 \
        --dumpProfile \
        --profileOutputFile "$profile" >/dev/null 2>"$dir/stderr_${iteration}_${warmup}.log" || rc=$?
    end=$(date +%s%N)
    local ms=$(( (end - start) / 1000000 ))
    write_record "$(emit_record "$condition" direct "$image" "$prompt_id" "$max_tokens" "$iteration" "$warmup" "$output" "$ms" null "$profile" "$([[ $rc -eq 0 ]] && echo true || echo false)" "$([[ $rc -eq 0 ]] && echo '' || echo "llm_inference exited $rc")")"
}

run_ipc() {
    local condition="$1" image="$2" prompt_id="$3" max_tokens="$4" iteration="$5" warmup="$6"
    local dir="${OUTPUT_DIR}/${condition}/ipc/$(basename "${image%.*}")"
    mkdir -p "$dir"
    local output="$dir/result_${iteration}_${warmup}.json"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[dry-run] ros2 run edge_vlm_ros vlm_single_shot_client --image $image"
        return
    fi

    local start end rc=0
    start=$(date +%s%N)
    ros2 run edge_vlm_ros vlm_single_shot_client \
        --socket "${EDGE_VLM_WORKER_SOCKET:-/tmp/edge_vlm.sock}" \
        --image "$image" \
        --prompt "${PROMPT_TEXT[$prompt_id]}" \
        --max-tokens "$max_tokens" \
        --output "$output" >/dev/null || rc=$?
    end=$(date +%s%N)
    local ms=$(( (end - start) / 1000000 ))
    write_record "$(emit_record "$condition" ipc "$image" "$prompt_id" "$max_tokens" "$iteration" "$warmup" "$output" null "$ms" "" "$([[ $rc -eq 0 ]] && echo true || echo false)" "$([[ $rc -eq 0 ]] && echo '' || echo "IPC client exited $rc")")"
}

export BENCH_RUN_ID="vlm_latency_${TIMESTAMP}"
IFS=',' read -r -a CONDITION_LIST <<< "$CONDITIONS"
IFS=',' read -r -a PATH_LIST <<< "$PATHS"

for condition in "${CONDITION_LIST[@]}"; do
    [[ -n "${CONDITION_PROMPT_ID[$condition]:-}" ]] || { echo "ERROR: unknown condition $condition" >&2; exit 1; }
    prompt_id="${CONDITION_PROMPT_ID[$condition]}"
    max_tokens="${CONDITION_MAX_TOKENS[$condition]}"
    for image in "${IMAGES[@]}"; do
        for path in "${PATH_LIST[@]}"; do
            for ((i=0; i<WARMUP; i++)); do
                "run_${path}" "$condition" "$image" "$prompt_id" "$max_tokens" "$i" true
            done
            for ((i=0; i<ITERATIONS; i++)); do
                "run_${path}" "$condition" "$image" "$prompt_id" "$max_tokens" "$i" false
            done
        done
    done
done

if [[ "${DRY_RUN}" == "false" ]]; then
    python3 "${SCRIPT_DIR}/vlm_latency_report.py" --input "$JSONL_OUT" --output "$REPORT_JSON" --text "$REPORT_TXT"
    echo "JSONL:  $JSONL_OUT"
    echo "Report: $REPORT_TXT"
fi
