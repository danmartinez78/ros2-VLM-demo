#!/usr/bin/env bash
set -Eeuo pipefail

edge_vlm_thor_yolo_hpcx_root="${EDGE_VLM_THOR_YOLO_HPCX_ROOT:-/opt/hpcx}"

edge_vlm_thor_yolo_fail() {
  echo "$1" >&2
  return 1 2>/dev/null || exit 1
}

edge_vlm_thor_yolo_find_lib_dir() {
  local lib_name="$1"
  local dir
  dir="$(find "${edge_vlm_thor_yolo_hpcx_root}" -maxdepth 4 -type f -name "${lib_name}" -printf '%h\n' 2>/dev/null | sort -u | head -n1 || true)"
  [[ -n "${dir}" ]] || edge_vlm_thor_yolo_fail "Missing ${lib_name} under ${edge_vlm_thor_yolo_hpcx_root}"
  printf '%s\n' "${dir}"
}

edge_vlm_thor_yolo_append_unique_path() {
  local candidate="$1"
  [[ -n "${candidate}" ]] || return 0
  case ":${EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH:-}:" in
    *:"${candidate}":*) ;;
    *)
      if [[ -n "${EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH:-}" ]]; then
        EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH="${EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH}:${candidate}"
      else
        EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH="${candidate}"
      fi
      ;;
  esac
}

ucx_lib_dir="$(edge_vlm_thor_yolo_find_lib_dir 'libucs.so*')"
ucc_lib_dir="$(edge_vlm_thor_yolo_find_lib_dir 'libucc.so*')"
ompi_lib_dir="$(edge_vlm_thor_yolo_find_lib_dir 'libmpi.so*' || true)"

edge_vlm_thor_yolo_append_unique_path "${ucx_lib_dir}"
edge_vlm_thor_yolo_append_unique_path "${ucc_lib_dir}"
edge_vlm_thor_yolo_append_unique_path "${ompi_lib_dir}"

if [[ -z "${EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH:-}" ]]; then
  edge_vlm_thor_yolo_fail "Failed to resolve NVIDIA HPC-X library paths under ${edge_vlm_thor_yolo_hpcx_root}"
fi

export EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
