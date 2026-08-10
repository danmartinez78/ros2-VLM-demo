#!/usr/bin/env bash
set -Eeuo pipefail

edge_vlm_thor_yolo_hpcx_root="${EDGE_VLM_THOR_YOLO_HPCX_ROOT:-/opt/hpcx}"

edge_vlm_thor_yolo_fail() {
  echo "$1" >&2
  return 1 2>/dev/null || exit 1
}

edge_vlm_thor_yolo_find_lib_dir() {
  local lib_name="$1"
  local required="${2:-required}"
  local preferred_subdir="${3:-}"
  local dir
  if [[ -n "${preferred_subdir}" ]]; then
    dir="${edge_vlm_thor_yolo_hpcx_root}/${preferred_subdir}"
    if compgen -G "${dir}/${lib_name}" > /dev/null; then
      printf '%s\n' "${dir}"
      return 0
    fi
  fi
  dir="$(find "${edge_vlm_thor_yolo_hpcx_root}" -maxdepth 4 -type f -name "${lib_name}" -printf '%h\n' 2>/dev/null | sort -u | head -n1 || true)"
  if [[ -z "${dir}" ]]; then
    if [[ "${required}" == "required" ]]; then
      edge_vlm_thor_yolo_fail "Missing ${lib_name} under ${edge_vlm_thor_yolo_hpcx_root}"
    fi
    return 0
  fi
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

ucx_lib_dir="$(edge_vlm_thor_yolo_find_lib_dir 'libucs.so*' required 'ucx/lib')"
ucc_lib_dir="$(edge_vlm_thor_yolo_find_lib_dir 'libucc.so*' required 'ucc/lib')"
ompi_lib_dir="$(edge_vlm_thor_yolo_find_lib_dir 'libmpi.so*' optional 'ompi/lib')"

edge_vlm_thor_yolo_append_unique_path "${ucx_lib_dir}"
edge_vlm_thor_yolo_append_unique_path "${ucc_lib_dir}"
edge_vlm_thor_yolo_append_unique_path "${ompi_lib_dir}"

if [[ -z "${EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH:-}" ]]; then
  edge_vlm_thor_yolo_fail "Failed to resolve NVIDIA HPC-X library paths under ${edge_vlm_thor_yolo_hpcx_root}"
fi

export EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH
resolved_ld_library_path="${EDGE_VLM_THOR_YOLO_HPCX_LD_LIBRARY_PATH}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  IFS=':' read -r -a edge_vlm_thor_yolo_existing_ld_paths <<< "${LD_LIBRARY_PATH}"
  for edge_vlm_thor_yolo_existing_ld_path in "${edge_vlm_thor_yolo_existing_ld_paths[@]}"; do
    case ":${resolved_ld_library_path}:" in
      *:"${edge_vlm_thor_yolo_existing_ld_path}":*) ;;
      *)
        resolved_ld_library_path="${resolved_ld_library_path}:${edge_vlm_thor_yolo_existing_ld_path}"
        ;;
    esac
  done
fi
export LD_LIBRARY_PATH="${resolved_ld_library_path}"
