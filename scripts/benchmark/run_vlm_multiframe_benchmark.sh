#!/usr/bin/env bash
# Generic multi-frame VLM latency benchmark.
#
# Sweeps selected frame counts over one ordered image sequence. The IPC path can
# exercise images, temporal_images, or video request contracts through the
# repository's vlm_multi_frame_client. The direct path remains available for
# ordered-image TensorRT Edge-LLM profiling.
set -euo pipefail

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/tmp/vlm_multiframe_bench_${TIMESTAMP}"
SEQUENCE_DIR=""
FRAME_COUNTS="1,2,4,8"
MAX_OUTPUT_TOKENS=32
WARMUP=1
ITERATIONS=3
PATHS="direct,ipc"
SEQUENCE_TYPE="images"
FPS=""
FRAME_TIMESTAMPS_SEC=""
RENDER_TIMESTAMPS=false
DRY_RUN=false
PROMPT_TEXT="Analyze this ordered temporal image sequence and describe the scene as compact JSON with keys: objects, actions, hazards, navigable. Be concise and provide one result for the full sequence."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence-dir) SEQUENCE_DIR="$2"; shift 2 ;;
        --frame-counts) FRAME_COUNTS="$2"; shift 2 ;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --paths) PATHS="$2"; shift 2 ;;
        --sequence-type) SEQUENCE_TYPE="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --frame-timestamps-sec) FRAME_TIMESTAMPS_SEC="$2"; shift 2 ;;
        --render-timestamps) RENDER_TIMESTAMPS=true; shift ;;
        --skip-ipc) PATHS="direct"; shift ;;
        --skip-direct) PATHS="ipc"; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done

[[ -n "$SEQUENCE_DIR" ]] || { echo "ERROR: --sequence-dir is required" >&2; exit 1; }
[[ -d "$SEQUENCE_DIR" ]] || { echo "ERROR: sequence directory does not exist: $SEQUENCE_DIR" >&2; exit 1; }
case "$SEQUENCE_TYPE" in images|temporal_images|video) ;; *) echo "ERROR: invalid --sequence-type" >&2; exit 1 ;; esac

mapfile -t ALL_FRAMES < <(find "$SEQUENCE_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort)
[[ ${#ALL_FRAMES[@]} -gt 0 ]] || { echo "ERROR: no JPEG/PNG frames found" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
JSONL_OUT="${OUTPUT_DIR}/vlm_multiframe_${TIMESTAMP}.jsonl"
REPORT_JSON="${OUTPUT_DIR}/vlm_multiframe_report.json"
REPORT_TXT="${OUTPUT_DIR}/vlm_multiframe_report.txt"
RUN_ID="vlm_multiframe_${TIMESTAMP}"
PROMPT_HASH=$(printf '%s' "$PROMPT_TEXT" | sha256sum | cut -c1-12)

select_frames() {
    python3 - "$1" "${ALL_FRAMES[@]}" <<'PY'
import sys
n = int(sys.argv[1]); frames = sys.argv[2:]
if len(frames) < n:
    raise SystemExit(f"need {n} frames, have {len(frames)}")
if n == 1:
    selected = [frames[0]]
elif n == len(frames):
    selected = frames
else:
    step = (len(frames) - 1) / (n - 1)
    selected = [frames[round(i * step)] for i in range(n)]
print("\n".join(selected))
PY
}

frame_metadata_json() {
    python3 - "$@" <<'PY'
import hashlib, json, sys
print(json.dumps([
    {"path": p, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    for p in sys.argv[1:]
]))
PY
}

write_record() {
    [[ "$DRY_RUN" == true ]] && { echo "[dry-run] record $1"; return; }
    printf '%s\n' "$1" >> "$JSONL_OUT"
}

make_record() {
    local condition="$1" path="$2" frame_meta="$3" iteration="$4" warmup="$5" result_path="$6" total_ms="$7" cold_ms="$8" success="$9" error="${10}" profile_path="${11:-}"
    python3 - "$RUN_ID" "$condition" "$path" "$frame_meta" "$PROMPT_HASH" "$MAX_OUTPUT_TOKENS" "$iteration" "$warmup" "$result_path" "$total_ms" "$cold_ms" "$success" "$error" "$SEQUENCE_TYPE" "$FPS" "$FRAME_TIMESTAMPS_SEC" "$RENDER_TIMESTAMPS" "$profile_path" <<'PY'
import json, os, sys
(run_id, condition, path, frame_meta, prompt_hash, max_tokens, iteration, warmup,
 result_path, total_ms, cold_ms, success, error, sequence_type, fps,
 timestamps, rendered, profile_path) = sys.argv[1:]
frames = json.loads(frame_meta)

def num(v): return None if v in ("", "null") else float(v)
rec = {
    "schema_version": "1", "record_type": "inference", "run_id": run_id,
    "recorded_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "frame_condition": condition, "frame_count": len(frames), "path": path,
    "frame_paths": frames, "prompt_hash": prompt_hash,
    "sequence_type": sequence_type,
    "fps": num(fps),
    "frame_timestamps_sec": [float(x) for x in timestamps.split(",") if x.strip()] or None,
    "frame_timestamp_policy": "explicit" if timestamps else ("fps" if fps else "none"),
    "rendered_timestamps": rendered == "true",
    "requested_sequence_type": sequence_type,
    "runtime_temporal_encoding": None,
    "temporal_fallback_used": None,
    "max_output_tokens": int(max_tokens), "actual_output_tokens": None,
    "total_image_tokens": None, "finish_reason": None, "output_text": None,
    "output_words": None, "inference_seconds": None,
    "success": success == "true", "error": error or None,
    "cold_start_total_ms": num(cold_ms), "total_latency_ms": num(total_ms),
    "ttft_ms": None, "vision_encoder_ms": None, "prefill_ms": None,
    "decode_ms": None, "decode_tokens_per_sec": None,
    "llm_generation_total_gpu_time_ms": None,
    "native_response_path": result_path if path == "direct" else None,
    "native_profile_path": profile_path or None,
    "ipc_result_path": result_path if path == "ipc" else None,
    "model_name": os.environ.get("EDGE_VLM_MODEL_NAME", "unknown"),
    "engine_provenance": {}, "iteration": int(iteration), "warmup": warmup == "true",
}
if result_path and os.path.isfile(result_path):
    try:
        data = json.load(open(result_path, encoding="utf-8"))
        rec["success"] = bool(data.get("success", rec["success"]))
        rec["output_text"] = data.get("output_text") or data.get("text")
        rec["actual_output_tokens"] = data.get("output_tokens") or data.get("generated_tokens")
        rec["output_words"] = data.get("output_words")
        rec["inference_seconds"] = data.get("inference_seconds")
        rec["requested_sequence_type"] = data.get("requested_sequence_type", rec["requested_sequence_type"])
        rec["runtime_temporal_encoding"] = data.get("runtime_temporal_encoding")
        rec["temporal_fallback_used"] = data.get("temporal_fallback_used")
        if data.get("error"): rec["error"] = data["error"]
    except Exception: pass
if profile_path and os.path.isfile(profile_path):
    try:
        p = json.load(open(profile_path, encoding="utf-8"))
        rec["total_image_tokens"] = p.get("multimodal", {}).get("total_image_tokens")
        g = p.get("generation", {})
        rec["actual_output_tokens"] = g.get("generated_tokens", rec["actual_output_tokens"])
        rec["decode_tokens_per_sec"] = g.get("tokens_per_second")
        rec["decode_ms"] = g.get("total_time_ms")
        rec["prefill_ms"] = p.get("prefill", {}).get("average_time_per_run_ms")
        for s in p.get("stages", []):
            sid = s.get("stage_id") or s.get("name")
            if sid == "vision_encoder": rec["vision_encoder_ms"] = s.get("average_time_per_run_ms")
            if sid == "llm_generation": rec["llm_generation_total_gpu_time_ms"] = s.get("total_gpu_time_ms")
    except Exception: pass
print(json.dumps(rec))
PY
}

build_direct_input() {
    local output="$1"; shift
    python3 - "$PROMPT_TEXT" "$output" "$@" <<'PY'
import json, sys
prompt, output, *frames = sys.argv[1:]
content = [{"type":"image", "image":p} for p in frames] + [{"type":"text", "text":prompt}]
json.dump({"requests":[{"messages":[{"role":"user","content":content}]}]}, open(output,"w"))
PY
}

run_ipc() {
    local condition="$1" iteration="$2" warmup="$3"; shift 3
    local frames=("$@") dir="$OUTPUT_DIR/$condition/ipc"; mkdir -p "$dir"
    local result="$dir/result_${iteration}_${warmup}.json"
    local args=(ros2 run edge_vlm_ros vlm_multi_frame_client --socket "${EDGE_VLM_WORKER_SOCKET:-/tmp/edge_vlm.sock}")
    for frame in "${frames[@]}"; do args+=(--image "$frame"); done
    args+=(--prompt "$PROMPT_TEXT" --max-tokens "$MAX_OUTPUT_TOKENS" --output "$result" --sequence-type "$SEQUENCE_TYPE")
    [[ -n "$FPS" ]] && args+=(--fps "$FPS")
    [[ -n "$FRAME_TIMESTAMPS_SEC" ]] && args+=(--frame-timestamps-sec "$FRAME_TIMESTAMPS_SEC")
    [[ "$RENDER_TIMESTAMPS" == true ]] && args+=(--render-timestamps)
    if [[ "$DRY_RUN" == true ]]; then printf '[dry-run] %q ' "${args[@]}"; echo; return; fi
    local start end rc=0; start=$(date +%s%N); "${args[@]}" >/dev/null || rc=$?; end=$(date +%s%N)
    local ms=$(( (end-start)/1000000 )) meta; meta=$(frame_metadata_json "${frames[@]}")
    write_record "$(make_record "$condition" ipc "$meta" "$iteration" "$warmup" "$result" "$ms" null "$([[ $rc -eq 0 ]] && echo true || echo false)" "$([[ $rc -eq 0 ]] && echo '' || echo "IPC client exited $rc")")"
}

run_direct() {
    local condition="$1" iteration="$2" warmup="$3"; shift 3
    local frames=("$@")
    if [[ "$SEQUENCE_TYPE" != images ]]; then
        echo "ERROR: direct benchmark currently supports --sequence-type=images only; use --paths ipc for native temporal/video contracts" >&2
        exit 2
    fi
    local binary="${TENSORRT_EDGE_LLM_ROOT:-}/build/examples/llm/llm_inference" dir="$OUTPUT_DIR/$condition/direct"; mkdir -p "$dir"
    local input="$dir/input_${iteration}_${warmup}.json" result="$dir/result_${iteration}_${warmup}.json" profile="$dir/profile_${iteration}_${warmup}.json"
    build_direct_input "$input" "${frames[@]}"
    if [[ "$DRY_RUN" == true ]]; then echo "[dry-run] $binary --inputFile $input"; return; fi
    local start end rc=0; start=$(date +%s%N)
    "$binary" --engineDir "${EDGE_VLM_LLM_ENGINE_DIR:?required}" --multimodalEngineDir "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:?required}" --maxGenerateLength "$MAX_OUTPUT_TOKENS" --inputFile "$input" --outputFile "$result" --warmup 0 --dumpProfile --profileOutputFile "$profile" >/dev/null 2>"$dir/stderr_${iteration}_${warmup}.log" || rc=$?
    end=$(date +%s%N); local ms=$(( (end-start)/1000000 )) meta; meta=$(frame_metadata_json "${frames[@]}")
    write_record "$(make_record "$condition" direct "$meta" "$iteration" "$warmup" "$result" null "$ms" "$([[ $rc -eq 0 ]] && echo true || echo false)" "$([[ $rc -eq 0 ]] && echo '' || echo "llm_inference exited $rc")" "$profile")"
}

IFS=',' read -r -a COUNTS <<< "$FRAME_COUNTS"
IFS=',' read -r -a PATH_LIST <<< "$PATHS"
for count in "${COUNTS[@]}"; do
    mapfile -t SELECTED < <(select_frames "$count")
    condition="F${count}"
    for path in "${PATH_LIST[@]}"; do
        for ((i=0; i<WARMUP; i++)); do "run_${path}" "$condition" "$i" true "${SELECTED[@]}"; done
        for ((i=0; i<ITERATIONS; i++)); do "run_${path}" "$condition" "$i" false "${SELECTED[@]}"; done
    done
done

if [[ "$DRY_RUN" == false ]]; then
    python3 "$SCRIPT_DIR/vlm_multiframe_report.py" --input "$JSONL_OUT" --output "$REPORT_JSON" --text "$REPORT_TXT"
    echo "JSONL:  $JSONL_OUT"
    echo "Report: $REPORT_TXT"
fi
