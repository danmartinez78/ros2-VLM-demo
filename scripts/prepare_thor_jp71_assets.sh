#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
manifest_path="${script_dir}/thor/jp71_manifest.json"
env_file="${EDGE_VLM_ENV_FILE:-${script_dir}/edge_vlm_env.sh}"

dry_run=0
skip_edge_llm=0
skip_model=0
skip_rtdetr=0
skip_data=0
model_name=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/prepare_thor_jp71_assets.sh [options]

Deterministic Thor JP7.1 asset preparation:
- clones/pins TensorRT-Edge-LLM to the tested revision,
- builds Edge-LLM plugin/runtime,
- prepares or validates model workspace layout,
- installs RT-DETR packages/models,
- prepares repo-owned test data assets,
- generates scripts/edge_vlm_env.sh from derived paths.

Options:
  --model-name <name>        Model profile name (default: Cosmos-Reason2-8B)
  --skip-edge-llm            Skip Edge-LLM clone/build phase
  --skip-model               Skip model workspace prep/validation phase
  --skip-rtdetr              Skip RT-DETR package/model install phase
  --skip-data                Skip rosbag + dataset preparation phase
  --dry-run                  Print planned actions without mutating the system
  -h, --help                 Show this help

Model acquisition knobs (for licensed/private assets):
  EDGE_VLM_MODEL_ARCHIVE             tar/tgz path or URL with prepared model directory
  EDGE_VLM_MODEL_ARCHIVE_SHA256      optional sha256 checksum for archive validation
  EDGE_VLM_MODEL_BUILD_COMMAND       command that prepares engine artifacts in workspace
USAGE
}

manifest_value() {
  local path="$1"
  python3 - "$manifest_path" "$path" <<'PY'
import json
import sys
manifest_path, path = sys.argv[1:3]
with open(manifest_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
value = data
for part in path.split('.'):
    value = value[part]
if isinstance(value, bool):
    print('true' if value else 'false')
elif value is None:
    print('')
else:
    print(value)
PY
}

manifest_list() {
  local path="$1"
  python3 - "$manifest_path" "$path" <<'PY'
import json
import sys
manifest_path, path = sys.argv[1:3]
with open(manifest_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
value = data
for part in path.split('.'):
    value = value[part]
if not isinstance(value, list):
    raise SystemExit(f"manifest path is not a list: {path}")
for item in value:
    print(item)
PY
}

manifest_list_optional() {
  local path="$1"
  python3 - "$manifest_path" "$path" <<'PY'
import json
import sys
manifest_path, path = sys.argv[1:3]
with open(manifest_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
value = data
for part in path.split('.'):
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        value = []
        break
if value is None:
    value = []
if not isinstance(value, list):
    raise SystemExit(f"manifest path is not a list: {path}")
for item in value:
    print(item)
PY
}

run_cmd() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf 'DRY-RUN  %s\n' "$*"
  else
    "$@"
  fi
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

infer_ros_workspace() {
  if [[ -n "${ROS_WORKSPACE:-}" ]]; then
    printf '%s\n' "${ROS_WORKSPACE}"
    return
  fi
  if [[ "$(basename -- "$(dirname -- "${repo_root}")")" == "src" ]]; then
    (cd -- "${repo_root}/../.." && pwd)
    return
  fi
  printf '%s\n' "${HOME}/ros2_ws"
}

validate_edge_artifacts() {
  local build_dir="$1"
  local missing=0
  [[ -d "${build_dir}" ]] || return 1
  while IFS= read -r rel; do
    [[ -f "${build_dir}/${rel}" ]] || missing=1
  done < <(manifest_list "edge_llm.required_build_artifacts")

  if [[ -f "${build_dir}/libNvInfer_edgellm_plugin.so" ]]; then
    :
  else
    local versioned
    versioned="$(find "${build_dir}" -maxdepth 1 -type f -name 'libNvInfer_edgellm_plugin.so.*' -print -quit || true)"
    if [[ -n "${versioned}" ]]; then
      run_cmd ln -sfn "$(basename -- "${versioned}")" "${build_dir}/libNvInfer_edgellm_plugin.so"
      missing=0
    else
      missing=1
    fi
  fi

  if [[ "${missing}" -ne 0 ]]; then
    return 1
  fi
}

validate_model_layout() {
  local workspace_dir="$1"
  local chosen_model="$2"
  local model_root="${workspace_dir}/${chosen_model}"
  local missing=0

  if [[ ! -d "${model_root}" ]]; then
    return 1
  fi

  while IFS= read -r artifact; do
    [[ -f "${model_root}/engine/llm/${artifact}" ]] || missing=1
  done < <(manifest_list_optional "models.${chosen_model}.required_llm_artifacts")

  while IFS= read -r artifact; do
    [[ -f "${model_root}/engine/${artifact}" ]] || missing=1
  done < <(manifest_list_optional "models.${chosen_model}.required_visual_artifacts")

  while IFS= read -r rel_dir; do
    [[ -d "${model_root}/${rel_dir}" ]] || missing=1
  done < <(manifest_list_optional "models.${chosen_model}.required_directories")

  [[ "${missing}" -eq 0 ]]
}

prepare_model_layout() {
  local workspace_dir="$1"
  local chosen_model="$2"
  local archive_source="${EDGE_VLM_MODEL_ARCHIVE:-}"
  local archive_sha256="${EDGE_VLM_MODEL_ARCHIVE_SHA256:-}"
  local prep_command="${EDGE_VLM_MODEL_BUILD_COMMAND:-}"
  local archive_file
  local stage_dir

  mkdir -p "${workspace_dir}"

  if validate_model_layout "${workspace_dir}" "${chosen_model}"; then
    echo "Model workspace already valid at ${workspace_dir}."
    return
  fi

  if [[ "${dry_run}" -eq 1 && -z "${archive_source}" && -z "${prep_command}" ]]; then
    echo "DRY-RUN  model workspace validation would fail without EDGE_VLM_MODEL_ARCHIVE or EDGE_VLM_MODEL_BUILD_COMMAND."
    return
  fi

  if [[ -n "${archive_source}" ]]; then
    stage_dir="$(mktemp -d /tmp/edge-vlm-model.XXXXXX)"
    archive_file="${stage_dir}/model.tar"
    trap 'rm -rf -- "${stage_dir}"' RETURN

    if [[ "${archive_source}" =~ ^https?:// ]]; then
      run_cmd curl -fL --retry 3 --retry-delay 2 -o "${archive_file}" "${archive_source}"
    else
      [[ -f "${archive_source}" ]] || fail "Model archive does not exist: ${archive_source}"
      run_cmd cp "${archive_source}" "${archive_file}"
    fi

    if [[ -n "${archive_sha256}" ]]; then
      if [[ "${dry_run}" -eq 1 ]]; then
        printf 'DRY-RUN  verify sha256 %s\n' "${archive_sha256}"
      else
        echo "${archive_sha256}  ${archive_file}" | sha256sum -c -
      fi
    fi

    run_cmd tar -xf "${archive_file}" -C "${workspace_dir}"
  fi

  if ! validate_model_layout "${workspace_dir}" "${chosen_model}"; then
    if [[ -n "${prep_command}" ]]; then
      if [[ "${dry_run}" -eq 1 ]]; then
        printf 'DRY-RUN  %s\n' "${prep_command}"
      else
        EDGE_VLM_WORKSPACE_DIR="${workspace_dir}" EDGE_VLM_MODEL_NAME="${chosen_model}" bash -lc "${prep_command}"
      fi
    fi
  fi

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "DRY-RUN  model layout validation complete."
  else
    validate_model_layout "${workspace_dir}" "${chosen_model}" || fail \
      "Model workspace is incomplete. Provide EDGE_VLM_MODEL_ARCHIVE or EDGE_VLM_MODEL_BUILD_COMMAND."
  fi
}

prepare_edge_llm() {
  local edge_root="$1"
  local edge_commit="$2"
  local edge_build="$3"

  if [[ -d "${edge_root}/.git" ]]; then
    run_cmd git -C "${edge_root}" fetch --tags --prune origin
  else
    run_cmd git clone "$(manifest_value edge_llm.repo_url)" "${edge_root}"
  fi

  run_cmd git -C "${edge_root}" checkout "${edge_commit}"
  run_cmd mkdir -p "${edge_build}"

  if ! validate_edge_artifacts "${edge_build}"; then
    run_cmd cmake -S "${edge_root}" -B "${edge_build}" -GNinja -DCMAKE_BUILD_TYPE=Release
    run_cmd cmake --build "${edge_build}" --parallel
  fi

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "DRY-RUN  edge build artifact validation complete."
  else
    validate_edge_artifacts "${edge_build}" || fail "Edge-LLM build artifacts are missing in ${edge_build}."
  fi
}

install_rtdetr_models() {
  local ros_distro="$1"
  local marker_file="${repo_root}/test_data/.rtdetr_models_install.ok"

  run_cmd sudo apt-get update
  run_cmd sudo apt-get install -y \
    "ros-${ros_distro}-isaac-ros-rtdetr" \
    "ros-${ros_distro}-isaac-ros-rtdetr-models-install"

  if [[ -f "${marker_file}" ]]; then
    echo "RT-DETR models installer already completed (marker: ${marker_file})."
    return
  fi

  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'DRY-RUN  source /opt/ros/%s/setup.bash && ros2 run isaac_ros_rtdetr_models_install install_rtdetr_models.sh --eula\n' "${ros_distro}"
  else
    # shellcheck disable=SC1090
    source "/opt/ros/${ros_distro}/setup.bash"
    ros2 run isaac_ros_rtdetr_models_install install_rtdetr_models.sh --eula
    mkdir -p "${repo_root}/test_data"
    date -u +%Y-%m-%dT%H:%M:%SZ >"${marker_file}"
  fi
}

generate_env_file() {
  local ros_distro="$1"
  local ros_workspace="$2"
  local edge_root="$3"
  local edge_build="$4"
  local chosen_model="$5"
  local workspace_dir="$6"
  local llm_dir="$7"
  local multimodal_dir="$8"
  local plugin_path="$9"

  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'DRY-RUN  generate %s\n' "${env_file}"
    return
  fi

  cat >"${env_file}" <<ENV
#!/usr/bin/env bash
# Generated by scripts/prepare_thor_jp71_assets.sh

export ROS_DISTRO="${ros_distro}"
export ROS_WORKSPACE="${ros_workspace}"
export TENSORRT_EDGE_LLM_ROOT="${edge_root}"
export TENSORRT_EDGE_LLM_BUILD_DIR="${edge_build}"
export EDGE_VLM_MODEL_NAME="${chosen_model}"
export EDGE_VLM_WORKSPACE_DIR="${workspace_dir}"
export EDGE_VLM_LLM_ENGINE_DIR="${llm_dir}"
export EDGE_VLM_MULTIMODAL_ENGINE_DIR="${multimodal_dir}"
export EDGELLM_PLUGIN_PATH="${plugin_path}"
ENV
  chmod +x "${env_file}"
}

for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  case "${arg}" in
    --model-name)
      next=$((i + 1))
      [[ "${next}" -le "$#" ]] || fail "Missing value for --model-name"
      model_name="${!next}"
      i=${next}
      ;;
    --skip-edge-llm) skip_edge_llm=1 ;;
    --skip-model) skip_model=1 ;;
    --skip-rtdetr) skip_rtdetr=1 ;;
    --skip-data) skip_data=1 ;;
    --dry-run) dry_run=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: ${arg}" ;;
  esac
done

[[ -f "${manifest_path}" ]] || fail "Manifest missing: ${manifest_path}"

default_model="$(manifest_value default_model)"
if [[ -z "${model_name}" ]]; then
  model_name="${EDGE_VLM_MODEL_NAME:-${default_model}}"
fi
[[ "${model_name}" == "Cosmos-Reason2-8B" ]] || fail "Unsupported --model-name '${model_name}' for this setup script."

ros_distro="${ROS_DISTRO:-jazzy}"
ros_workspace="$(infer_ros_workspace)"
default_edge_root="$(manifest_value edge_llm.default_root)"
default_edge_build="$(manifest_value edge_llm.default_build_dir)"
default_workspace="$(manifest_value default_workspace)"

edge_root="${TENSORRT_EDGE_LLM_ROOT:-${default_edge_root}}"
edge_root="${edge_root/\$\{HOME\}/${HOME}}"
edge_build="${TENSORRT_EDGE_LLM_BUILD_DIR:-${default_edge_build}}"
edge_build="${edge_build/\$\{HOME\}/${HOME}}"
workspace_dir="${EDGE_VLM_WORKSPACE_DIR:-${default_workspace}}"
workspace_dir="${workspace_dir/\$\{HOME\}/${HOME}}"
llm_dir="${EDGE_VLM_LLM_ENGINE_DIR:-${workspace_dir}/${model_name}/engine/llm}"
multimodal_dir="${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-${workspace_dir}/${model_name}/engine}"
plugin_path="${EDGELLM_PLUGIN_PATH:-${edge_build}/libNvInfer_edgellm_plugin.so}"

printf 'Thor JP7.1 setup plan:\n'
printf '  ROS_DISTRO: %s\n' "${ros_distro}"
printf '  ROS_WORKSPACE: %s\n' "${ros_workspace}"
printf '  Edge-LLM root: %s\n' "${edge_root}"
printf '  Edge-LLM build: %s\n' "${edge_build}"
printf '  Edge-LLM commit: %s\n' "$(manifest_value edge_llm.commit)"
printf '  Model: %s\n' "${model_name}"
printf '  Model workspace: %s\n' "${workspace_dir}"
printf '  Env file: %s\n' "${env_file}"

if [[ "${skip_edge_llm}" -eq 0 ]]; then
  prepare_edge_llm "${edge_root}" "$(manifest_value edge_llm.commit)" "${edge_build}"
fi

if [[ "${skip_model}" -eq 0 ]]; then
  prepare_model_layout "${workspace_dir}" "${model_name}"
fi

if [[ "${skip_rtdetr}" -eq 0 ]]; then
  install_rtdetr_models "${ros_distro}"
fi

if [[ "${skip_data}" -eq 0 ]]; then
  run_cmd bash "${script_dir}/test_data/download_rosbags.sh" download image-proc
  run_cmd bash "${script_dir}/test_data/download_rosbags.sh" download h264
  run_cmd bash "${script_dir}/test_data/download_rosbags.sh" download nvblox
  run_cmd bash "${script_dir}/test_data/download_rosbags.sh" download rtdetr
  run_cmd bash "${script_dir}/test_data/prepare_datasets.sh"
fi

generate_env_file \
  "${ros_distro}" \
  "${ros_workspace}" \
  "${edge_root}" \
  "${edge_build}" \
  "${model_name}" \
  "${workspace_dir}" \
  "${llm_dir}" \
  "${multimodal_dir}" \
  "${plugin_path}"

echo "Preparation completed."
