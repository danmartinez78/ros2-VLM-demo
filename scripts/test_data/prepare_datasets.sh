#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
manifest_path="${script_dir}/manifests/assets_manifest.json"
datasets_root="${repo_root}/test_data/datasets"

usage() {
  cat <<'USAGE'
Usage: prepare_datasets.sh [--dry-run]

Prepares deterministic dataset layout expected by tests/benchmarks:
- test_data/datasets/jaad
- test_data/datasets/nuscenes-mini

Automatic downloads are performed only where licenses allow direct access.
For JAAD clips and nuScenes mini, provide archives via:
  JAAD_CLIPS_ARCHIVE            local tar/zip or URL
  NUSCENES_MINI_ARCHIVE         local tar/zip or URL
USAGE
}

dry_run=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

run_cmd() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf 'DRY-RUN  %s\n' "$*"
  else
    "$@"
  fi
}

extract_archive() {
  local source="$1"
  local dest="$2"
  local stage="$(mktemp -d /tmp/edge-vlm-dataset.XXXXXX)"
  local file="${stage}/archive"
  trap 'rm -rf -- "${stage}"' RETURN

  if [[ "${source}" =~ ^https?:// ]]; then
    run_cmd curl -fL --retry 3 --retry-delay 2 -o "${file}" "${source}"
  else
    [[ -f "${source}" ]] || { echo "Archive not found: ${source}" >&2; return 1; }
    run_cmd cp "${source}" "${file}"
  fi

  run_cmd mkdir -p "${dest}"
  if [[ "${source}" == *.zip ]]; then
    run_cmd unzip -o "${file}" -d "${dest}"
  else
    run_cmd tar -xf "${file}" -C "${dest}"
  fi
}

run_cmd mkdir -p "${datasets_root}"

jaad_root="${datasets_root}/jaad"
nuscenes_root="${datasets_root}/nuscenes-mini"

run_cmd mkdir -p "${jaad_root}/prepared/contact_sheets"
run_cmd mkdir -p "${jaad_root}/clips"
if [[ ! -d "${jaad_root}/annotations/.git" ]]; then
  run_cmd git clone --depth 1 https://github.com/ykotseruba/JAAD.git "${jaad_root}/annotations"
fi

if [[ -n "${JAAD_CLIPS_ARCHIVE:-}" && ! -f "${jaad_root}/.clips-ready" ]]; then
  extract_archive "${JAAD_CLIPS_ARCHIVE}" "${jaad_root}/clips"
  if [[ "$dry_run" -eq 0 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ >"${jaad_root}/.clips-ready"
  fi
fi

run_cmd mkdir -p "${nuscenes_root}/prepared/sequences"
if [[ -n "${NUSCENES_MINI_ARCHIVE:-}" && ! -f "${nuscenes_root}/.mini-ready" ]]; then
  extract_archive "${NUSCENES_MINI_ARCHIVE}" "${nuscenes_root}"
  if [[ "$dry_run" -eq 0 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ >"${nuscenes_root}/.mini-ready"
  fi
fi

missing=0
for required in \
  "${jaad_root}/annotations" \
  "${jaad_root}/clips" \
  "${jaad_root}/prepared/contact_sheets" \
  "${nuscenes_root}/samples" \
  "${nuscenes_root}/sweeps" \
  "${nuscenes_root}/maps" \
  "${nuscenes_root}/prepared/sequences"; do
  if [[ ! -e "${required}" ]]; then
    echo "MISSING  ${required}"
    missing=1
  fi
done

if [[ "${dry_run}" -eq 1 ]]; then
  echo "DRY-RUN  dataset validation checks complete."
  exit 0
fi

if [[ "${missing}" -ne 0 ]]; then
  cat >&2 <<'EOF_MSG'
Dataset preparation is incomplete.
- JAAD clips require accepted JAAD licensing and credentials.
- nuScenes mini requires accepted nuScenes licensing and account access.
Provide JAAD_CLIPS_ARCHIVE and NUSCENES_MINI_ARCHIVE, then rerun.
EOF_MSG
  exit 1
fi

echo "Dataset layout ready under ${datasets_root}."
