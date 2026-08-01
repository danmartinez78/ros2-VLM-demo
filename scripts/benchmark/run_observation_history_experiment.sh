#!/usr/bin/env bash
# Compare single-frame inference with 0, 1, and 3 retained semantic observations.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
history_entries_csv="0,1,3"
success_results=4
playback_duration=30
result_timeout=180
max_generate_length=96
history_max_chars=1500
output_dir=""
resume=false

usage() {
  cat <<'EOF'
Usage: run_observation_history_experiment.sh [options]

Options:
  --output-dir DIR          Artifact root (default: timestamped directory under /tmp)
  --history-entries CSV     Observation counts to compare (default: 0,1,3)
  --success-results N       Successful frames required per run (default: 4)
  --playback-duration N     Maximum bag playback seconds per run (default: 30)
  --result-timeout N        Maximum result wait seconds per run (default: 180)
  --max-generate-length N   Output token limit (default: 96)
  --history-max-chars N     Observation-history character budget (default: 1500)
  --resume                  Skip configurations with complete summaries
  --help                    Show this help
EOF
}

while (($#)); do
  case "$1" in
    --output-dir) output_dir="$2"; shift 2 ;;
    --history-entries) history_entries_csv="$2"; shift 2 ;;
    --success-results) success_results="$2"; shift 2 ;;
    --playback-duration) playback_duration="$2"; shift 2 ;;
    --result-timeout) result_timeout="$2"; shift 2 ;;
    --max-generate-length) max_generate_length="$2"; shift 2 ;;
    --history-max-chars) history_max_chars="$2"; shift 2 ;;
    --resume) resume=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in success_results playback_duration result_timeout max_generate_length; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${value_name} must be a positive integer; got '${value}'." >&2
    exit 2
  }
done
[[ "${history_max_chars}" =~ ^[0-9]+$ ]] || {
  echo "history_max_chars must be a non-negative integer." >&2
  exit 2
}

IFS=',' read -r -a history_entries <<< "${history_entries_csv}"
(("${#history_entries[@]}" > 0)) || { echo "No history configurations supplied." >&2; exit 2; }
for entries in "${history_entries[@]}"; do
  [[ "${entries}" =~ ^[0-9]+$ ]] || {
    echo "Invalid history entry count: '${entries}'." >&2
    exit 2
  }
done

if [[ -z "${output_dir}" ]]; then
  output_dir="/tmp/cosmos_observation_history_$(date -u +%Y%m%d_%H%M%S)"
fi
mkdir -p "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd)"

system_instruction="You are a robot vision observer. Prior observations are unverified context and may be wrong. Base object and hazard claims on the current image. Mention change only when the current image supports it. Respond concisely."
test_prompt="Report visible people, vehicles, robots, obstacles, and immediate hazards. Briefly note meaningful differences from prior observations when supported."

echo "Observation-history experiment artifacts: ${output_dir}"
echo "Configurations: ${history_entries_csv}"

for entries in "${history_entries[@]}"; do
  run_dir="${output_dir}/history_${entries}"
  echo
  if [[ "${resume}" == true && -s "${run_dir}/manifest.json" &&
        -s "${run_dir}/result_summary.json" && -s "${run_dir}/ros_metrics.json" ]]; then
    echo "==> Reusing completed run with ${entries} retained observation(s)"
    continue
  fi
  echo "==> Running with ${entries} retained observation(s)"
  PLAYBACK_DURATION_SECONDS="${playback_duration}" \
  RESULT_TIMEOUT_SECONDS="${result_timeout}" \
  MAX_GENERATE_LENGTH="${max_generate_length}" \
  SUCCESS_RESULTS_REQUIRED="${success_results}" \
  INSTRUCTION_DELIVERY_MODE=structured \
  OBSERVATION_HISTORY_MAX_ENTRIES="${entries}" \
  OBSERVATION_HISTORY_MAX_CHARS="${history_max_chars}" \
  SYSTEM_INSTRUCTION="${system_instruction}" \
  TEST_PROMPT="${test_prompt}" \
  ARTIFACT_DIR="${run_dir}" \
    bash "${repo_root}/scripts/test_data/run_image_proc_test.sh"

  python3 "${script_dir}/summarize_observation_history.py" \
    --input "${run_dir}/results.log" \
    --output "${run_dir}/result_summary.json"

  if [[ -s "${run_dir}/benchmark.jsonl" ]]; then
    python3 "${script_dir}/collect_ros_metrics.py" \
      --input "${run_dir}/benchmark.jsonl" \
      --warmup 0 \
      --output "${run_dir}/ros_metrics.json"
  fi
done

python3 - "${output_dir}" "${history_entries_csv}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
entries = [int(value) for value in sys.argv[2].split(",")]
runs = []
for count in entries:
    run_dir = root / f"history_{count}"
    runs.append(
        {
            "observation_history_max_entries": count,
            "directory": str(run_dir),
            "manifest": json.loads((run_dir / "manifest.json").read_text()),
            "result_summary": json.loads((run_dir / "result_summary.json").read_text()),
            "ros_metrics": (
                json.loads((run_dir / "ros_metrics.json").read_text())
                if (run_dir / "ros_metrics.json").exists()
                else None
            ),
        }
    )
manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "purpose": "single-frame observation-history comparison",
    "runs": runs,
}
(root / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

echo
echo "Experiment complete."
echo "Combined manifest: ${output_dir}/experiment.json"
