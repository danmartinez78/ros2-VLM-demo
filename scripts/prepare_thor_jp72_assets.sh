#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
manifest_path="${script_dir}/thor/jp72_manifest.json"
env_file="${EDGE_VLM_ENV_FILE:-${script_dir}/edge_vlm_env.sh}"
source "${script_dir}/ros_setup_guard.sh"

dry_run=0
skip_edge_llm=0
skip_model=0
skip_rtdetr=0
skip_data=0
model_name=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/prepare_thor_jp72_assets.sh [options]

Deterministic Thor JP7.2 asset preparation:
- clones/pins TensorRT-Edge-LLM to the tested revision,
- builds Edge-LLM plugin/runtime with Thor CMake target settings,
- prepares Cosmos-Reason2-8B quantized/onnx/engine stages,
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

Optional model override knobs (fallback path):
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

source "${script_dir}/apt_transaction_guard.sh"

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

resolve_hf_token() {
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    printf '%s\n' "${HUGGING_FACE_HUB_TOKEN}"
    return
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s\n' "${HF_TOKEN}"
    return
  fi
  if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    head -n 1 "${HOME}/.cache/huggingface/token"
    return
  fi
  if [[ -f "${HOME}/.huggingface/token" ]]; then
    head -n 1 "${HOME}/.huggingface/token"
    return
  fi
  printf '\n'
}

preflight_cosmos_hf_access() {
  local hf_model_id="$1"
  local token
  token="$(resolve_hf_token)"

  if [[ "${dry_run}" -eq 1 ]]; then
    if [[ -n "${token}" ]]; then
      printf 'DRY-RUN  verify Hugging Face access for %s using cached token\n' "${hf_model_id}"
    else
      printf 'DRY-RUN  verify Hugging Face access for %s (token required: run huggingface-cli login and accept model license)\n' "${hf_model_id}"
    fi
    return
  fi

  if [[ -z "${token}" ]]; then
    fail "Missing Hugging Face credentials for ${hf_model_id}. Accept the license at https://huggingface.co/${hf_model_id} and run 'huggingface-cli login' (or set HUGGING_FACE_HUB_TOKEN)."
  fi

  local status
  local auth_header="Authorization: Bearer ${token}"
  status="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      -H "${auth_header}" \
      "https://huggingface.co/${hf_model_id}/resolve/main/config.json"
  )"

  if [[ "${status}" == "200" ]]; then
    return
  fi

  if [[ "${status}" == "401" || "${status}" == "403" ]]; then
    fail "Hugging Face access denied for ${hf_model_id} (HTTP ${status}). Accept the gated license and ensure your token has read access."
  fi

  fail "Unable to verify Hugging Face access for ${hf_model_id} (HTTP ${status})."
}

detect_cuda_ctk_version() {
  nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
}

validate_thor_build_inputs() {
  local edge_root="$1"
  local expected_cuda_ctk="$2"
  local detected_cuda_ctk=""

  if command -v nvcc >/dev/null 2>&1; then
    detected_cuda_ctk="$(detect_cuda_ctk_version)"
  fi

  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'DRY-RUN  validate Thor build inputs (nvcc, TensorRT headers, toolchain file)\n'
    if [[ -n "${detected_cuda_ctk}" ]]; then
      printf 'DRY-RUN  detected CUDA toolkit: %s (configure target: %s)\n' "${detected_cuda_ctk}" "${expected_cuda_ctk}"
    else
      printf 'DRY-RUN  CUDA toolkit detection deferred to runtime host\n'
    fi
    return
  fi

  command -v nvcc >/dev/null 2>&1 || fail "nvcc not found. Install the JP7.2 CUDA toolkit components before building Edge-LLM."
  [[ -n "${detected_cuda_ctk}" ]] || fail "Unable to detect CUDA toolkit version from nvcc."
  [[ "${detected_cuda_ctk}" == "${expected_cuda_ctk}" ]] || fail \
    "Detected CUDA toolkit ${detected_cuda_ctk}, but Thor profile requires ${expected_cuda_ctk}. Use the JP7.2-supported stack on this host (no auto-upgrade is performed by this script)."
  [[ -f "${edge_root}/cmake/aarch64_linux_toolchain.cmake" ]] || fail \
    "Missing TensorRT-Edge-LLM toolchain file: ${edge_root}/cmake/aarch64_linux_toolchain.cmake"
  [[ -f /usr/include/NvInfer.h || -f /usr/include/aarch64-linux-gnu/NvInfer.h ]] || fail \
    "TensorRT development headers not found under /usr/include. Install JP7.2 TensorRT dev packages before building."
}

prepare_edge_llm() {
  local edge_root="$1"
  local edge_commit="$2"
  local edge_build="$3"
  local cuda_ctk_version="$4"

  if [[ -d "${edge_root}/.git" ]]; then
    run_cmd git -C "${edge_root}" fetch --tags --prune origin
  else
    run_cmd git clone "$(manifest_value edge_llm.repo_url)" "${edge_root}"
  fi

  run_cmd git -C "${edge_root}" checkout "${edge_commit}"
  run_cmd git -C "${edge_root}" submodule update --init --recursive
  run_cmd mkdir -p "${edge_build}"

  if ! validate_edge_artifacts "${edge_build}"; then
    validate_thor_build_inputs "${edge_root}" "${cuda_ctk_version}"
    run_cmd cmake -S "${edge_root}" -B "${edge_build}" \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DTRT_PACKAGE_DIR=/usr \
      -DCMAKE_TOOLCHAIN_FILE="${edge_root}/cmake/aarch64_linux_toolchain.cmake" \
      -DEMBEDDED_TARGET=jetson-thor \
      -DCUDA_CTK_VERSION="${cuda_ctk_version}" \
      -DENABLE_CUTE_DSL=ALL
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
  local isaac_ros_ws="$2"
  local ros_setup="${EDGE_VLM_ROS_SETUP_PATH:-/opt/ros/${ros_distro}/setup.bash}"
  local marker_file="${EDGE_VLM_RTDETR_MARKER_FILE:-${repo_root}/test_data/.rtdetr_models_install.ok}"
  local -a rtdetr_packages=(
    "ros-${ros_distro}-isaac-ros-rtdetr"
    "ros-${ros_distro}-isaac-ros-rtdetr-models-install"
  )

  if [[ "${EDGE_VLM_APT_GUARD_TEST_MODE:-0}" == "1" ]]; then
    [[ -n "${EDGE_VLM_APT_SIMULATION_OUTPUT:-}" ]] || fail \
      "EDGE_VLM_APT_GUARD_TEST_MODE requires EDGE_VLM_APT_SIMULATION_OUTPUT."
    assert_safe_apt_transaction "RT-DETR packages" "${rtdetr_packages[@]}"
    echo "APT guard test transaction passed for RT-DETR packages."
    return
  fi

  if [[ -f "${marker_file}" ]]; then
    echo "RT-DETR models installer already completed (marker: ${marker_file})."
    return
  fi

  run_cmd mkdir -p "${isaac_ros_ws}/src"

  if [[ "${dry_run}" -ne 1 ]]; then
    assert_safe_apt_transaction "RT-DETR packages" "${rtdetr_packages[@]}"
  fi
  run_cmd sudo apt-get update
  run_cmd sudo apt-get install -y "${rtdetr_packages[@]}"

  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'DRY-RUN  source %s && ISAAC_ROS_WS=%s ros2 run isaac_ros_rtdetr_models_install install_rtdetr_models.sh --eula\n' "${ros_setup}" "${isaac_ros_ws}"
  else
    source_ros_setup_nounset_safe "${ros_setup}" || fail \
      "Unable to source ${ros_setup}."
    env "ISAAC_ROS_WS=${isaac_ros_ws}" ros2 run isaac_ros_rtdetr_models_install install_rtdetr_models.sh --eula
    mkdir -p "$(dirname -- "${marker_file}")"
    date -u +%Y-%m-%dT%H:%M:%SZ >"${marker_file}"
  fi
}

apply_model_overrides() {
  local workspace_dir="$1"
  local chosen_model="$2"
  local archive_source="${EDGE_VLM_MODEL_ARCHIVE:-}"
  local archive_sha256="${EDGE_VLM_MODEL_ARCHIVE_SHA256:-}"
  local prep_command="${EDGE_VLM_MODEL_BUILD_COMMAND:-}"
  local archive_file
  local stage_dir

  [[ -n "${archive_source}" || -n "${prep_command}" ]] || return 0

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "DRY-RUN  model override inputs detected; showing override/fallback actions first."
  fi

  run_cmd mkdir -p "${workspace_dir}"

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

  if ! validate_model_layout "${workspace_dir}" "${chosen_model}" && [[ -n "${prep_command}" ]]; then
    if [[ "${dry_run}" -eq 1 ]]; then
      printf 'DRY-RUN  %s\n' "${prep_command}"
    else
      EDGE_VLM_WORKSPACE_DIR="${workspace_dir}" EDGE_VLM_MODEL_NAME="${chosen_model}" bash -lc "${prep_command}"
    fi
  fi
}

is_quantized_ready() {
  local model_root="$1"
  local quantized_dir="${model_root}/quantized"
  [[ -d "${quantized_dir}" ]] || return 1
  find "${quantized_dir}" -type f \( -name '*.safetensors' -o -name '*.json' \) -print -quit | grep -q .
}

is_onnx_ready() {
  local model_root="$1"
  local llm_onnx_dir="${model_root}/onnx/llm"
  local visual_onnx_dir="${model_root}/onnx/visual"
  [[ -d "${llm_onnx_dir}" && -d "${visual_onnx_dir}" ]] || return 1
  find "${llm_onnx_dir}" -type f -name '*.onnx' -print -quit | grep -q . && \
    find "${visual_onnx_dir}" -type f -name '*.onnx' -print -quit | grep -q .
}

ensure_docker_available() {
  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'DRY-RUN  validate Docker + NVIDIA runtime availability\n'
    return
  fi
  command -v docker >/dev/null 2>&1 || fail "docker is required for Cosmos export/quantization stage."
}

run_cosmos_container_stage() {
  local container_image="$1"
  local edge_root="$2"
  local workspace_dir="$3"
  local model_name_local="$4"
  local script_body="$5"
  local hf_token="$6"
  local hf_token_temp_dir=""

  ensure_docker_available
  run_cmd docker pull "${container_image}"

  local -a docker_args=(
    docker run --rm --runtime nvidia --gpus all
    -v "${edge_root}:${edge_root}"
    -v "${workspace_dir}:${workspace_dir}"
    -e "HOME=${HOME}"
    -e "HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}"
    -e "MODEL_NAME=${model_name_local}"
    -e "WORKSPACE_DIR=${workspace_dir}"
    -w "${workspace_dir}"
  )

  if [[ -d "${HOME}/.cache/huggingface" ]]; then
    docker_args+=( -v "${HOME}/.cache/huggingface:${HOME}/.cache/huggingface" )
  fi
  if [[ -d "${HOME}/.huggingface" ]]; then
    docker_args+=( -v "${HOME}/.huggingface:${HOME}/.huggingface" )
  fi

  if [[ -n "${hf_token}" ]] && [[ ! -f "${HOME}/.cache/huggingface/token" ]] && [[ ! -f "${HOME}/.huggingface/token" ]]; then
    hf_token_temp_dir="$(mktemp -d /tmp/edge-vlm-hf-token.XXXXXX)"
    printf '%s\n' "${hf_token}" >"${hf_token_temp_dir}/token"
    chmod 600 "${hf_token_temp_dir}/token"
    docker_args+=( -v "${hf_token_temp_dir}/token:${HOME}/.cache/huggingface/token:ro" )
  fi

  docker_args+=( "${container_image}" bash -lc "${script_body}" )
  run_cmd "${docker_args[@]}"

  if [[ -n "${hf_token_temp_dir}" ]]; then
    rm -rf -- "${hf_token_temp_dir}"
  fi
}

build_cosmos_engines() {
  local model_root="$1"
  local edge_build="$2"
  local plugin_path="$3"

  local llm_builder="${edge_build}/examples/llm/llm_build"
  local visual_builder="${edge_build}/examples/multimodal/visual_build"
  local plugin_dir=""

  plugin_dir="$(cd -- "$(dirname -- "${plugin_path}")" && pwd)"
  plugin_path="${plugin_dir}/$(basename -- "${plugin_path}")"

  if [[ "${dry_run}" -ne 1 ]]; then
    [[ -x "${llm_builder}" ]] || fail "Missing llm_build executable at ${llm_builder}. Build Edge-LLM first."
    [[ -x "${visual_builder}" ]] || fail "Missing visual_build executable at ${visual_builder}. Build Edge-LLM first."
    [[ -f "${plugin_path}" ]] || fail \
      "Missing Edge-LLM plugin at ${plugin_path}. Build TensorRT-Edge-LLM and ensure EDGELLM_PLUGIN_PATH points to libNvInfer_edgellm_plugin.so."
  fi

  run_cmd mkdir -p "${model_root}/engine/llm" "${model_root}/engine"
  run_cmd env "EDGELLM_PLUGIN_PATH=${plugin_path}" "${llm_builder}" \
    --onnxDir "${model_root}/onnx/llm" \
    --engineDir "${model_root}/engine/llm" \
    --maxBatchSize "${EDGE_VLM_LLM_MAX_BATCH_SIZE:-1}" \
    --maxInputLen "${EDGE_VLM_LLM_MAX_INPUT_LEN:-1024}" \
    --maxKVCacheCapacity "${EDGE_VLM_LLM_MAX_KV_CACHE_CAPACITY:-4096}"
  run_cmd env "EDGELLM_PLUGIN_PATH=${plugin_path}" "${visual_builder}" \
    --onnxDir "${model_root}/onnx/visual" \
    --engineDir "${model_root}/engine"
}

prepare_cosmos_default() {
  local workspace_dir="$1"
  local chosen_model="$2"
  local edge_root="$3"
  local edge_build="$4"
  local model_root="${workspace_dir}/${chosen_model}"
  local hf_model_id
  local quantization
  local container_image
  local token
  local quantized_ready
  local onnx_ready
  local engine_ready
  local plan_quantize
  local plan_export
  local plan_engine_build
  local container_preamble
  local plugin_path="$5"
  local modelopt_version

  hf_model_id="$(manifest_value "models.${chosen_model}.hf_model_id")"
  quantization="$(manifest_value "models.${chosen_model}.quantization")"
  container_image="$(manifest_value "models.${chosen_model}.pytorch_container")"
  modelopt_version="$(manifest_value "models.${chosen_model}.modelopt_version")"

  quantized_ready=0
  onnx_ready=0
  engine_ready=0

  if is_quantized_ready "${model_root}"; then
    quantized_ready=1
  fi
  if is_onnx_ready "${model_root}"; then
    onnx_ready=1
  fi
  if validate_model_layout "${workspace_dir}" "${chosen_model}"; then
    engine_ready=1
  fi

  printf 'Cosmos stage status (%s): quantized=%s onnx=%s engines=%s\n' \
    "${chosen_model}" "${quantized_ready}" "${onnx_ready}" "${engine_ready}"

  plan_quantize=0
  plan_export=0
  plan_engine_build=0
  if [[ "${quantized_ready}" -eq 0 ]]; then
    plan_quantize=1
    onnx_ready=0
    engine_ready=0
  fi
  if [[ "${onnx_ready}" -eq 0 ]]; then
    plan_export=1
    engine_ready=0
  fi
  if [[ "${engine_ready}" -eq 0 ]]; then
    plan_engine_build=1
  fi

  if [[ "${plan_quantize}" -eq 1 ]]; then
    preflight_cosmos_hf_access "${hf_model_id}"
  fi
  container_preamble="set -Eeuo pipefail; cd '${edge_root}'; python3 -m venv --system-site-packages /tmp/edgellm-venv; source /tmp/edgellm-venv/bin/activate; pip3 install --no-deps .; sed '/^torch/d' requirements.txt > /tmp/edge-llm-reqs-no-torch.txt; pip3 install -r /tmp/edge-llm-reqs-no-torch.txt; pip3 install --no-deps 'nvidia-modelopt==${modelopt_version}'; cd '${workspace_dir}'; mkdir -p '${chosen_model}'"

  if [[ "${dry_run}" -eq 1 ]]; then
    if [[ "${plan_quantize}" -eq 1 || "${plan_export}" -eq 1 ]]; then
      printf 'DRY-RUN  planned stage: docker pull %s\n' "${container_image}"
    fi
    if [[ "${plan_quantize}" -eq 1 ]]; then
      printf 'DRY-RUN  planned stage: ensure nvidia-modelopt==%s inside Edge-LLM venv\n' "${modelopt_version}"
      printf 'DRY-RUN  planned stage: docker run %s ... tensorrt-edgellm-quantize llm --model_dir %s --output_dir %s/quantized --quantization %s\n' \
        "${container_image}" "${hf_model_id}" "${chosen_model}" "${quantization}"
    fi
    if [[ "${plan_export}" -eq 1 ]]; then
      printf 'DRY-RUN  planned stage: docker run %s ... tensorrt-edgellm-export %s/quantized %s/onnx\n' \
        "${container_image}" "${chosen_model}" "${chosen_model}"
    fi
    if [[ "${plan_engine_build}" -eq 1 ]]; then
      printf 'DRY-RUN  planned stage: EDGELLM_PLUGIN_PATH=%s native Thor llm_build --onnxDir %s/onnx/llm --engineDir %s/engine/llm\n' \
        "${plugin_path}" "${model_root}" "${model_root}"
      printf 'DRY-RUN  planned stage: EDGELLM_PLUGIN_PATH=%s native Thor visual_build --onnxDir %s/onnx/visual --engineDir %s/engine\n' \
        "${plugin_path}" "${model_root}" "${model_root}"
    fi
    return
  fi

  token=""
  if [[ "${plan_quantize}" -eq 1 || "${plan_export}" -eq 1 ]]; then
    token="$(resolve_hf_token)"
  fi
  run_cmd mkdir -p "${workspace_dir}" "${model_root}"

  if [[ "${quantized_ready}" -eq 0 ]]; then
    run_cosmos_container_stage \
      "${container_image}" \
      "${edge_root}" \
      "${workspace_dir}" \
      "${chosen_model}" \
      "${container_preamble}; tensorrt-edgellm-quantize llm --model_dir '${hf_model_id}' --output_dir '${chosen_model}/quantized' --quantization '${quantization}'" \
      "${token}"
    quantized_ready=1
    onnx_ready=0
    engine_ready=0
  fi

  if [[ "${onnx_ready}" -eq 0 ]]; then
    run_cosmos_container_stage \
      "${container_image}" \
      "${edge_root}" \
      "${workspace_dir}" \
      "${chosen_model}" \
      "${container_preamble}; tensorrt-edgellm-export '${chosen_model}/quantized' '${chosen_model}/onnx'" \
      "${token}"
    onnx_ready=1
    engine_ready=0
  fi

  if [[ "${engine_ready}" -eq 0 ]]; then
    build_cosmos_engines "${model_root}" "${edge_build}" "${plugin_path}"
  fi

  validate_model_layout "${workspace_dir}" "${chosen_model}" || fail \
    "Model workspace is incomplete after first-class Cosmos preparation."
}

prepare_model_layout() {
  local workspace_dir="$1"
  local chosen_model="$2"
  local edge_root="$3"
  local edge_build="$4"
  local plugin_path="$5"

  if validate_model_layout "${workspace_dir}" "${chosen_model}"; then
    echo "Model workspace already valid at ${workspace_dir}."
    return
  fi

  apply_model_overrides "${workspace_dir}" "${chosen_model}"
  if validate_model_layout "${workspace_dir}" "${chosen_model}"; then
    echo "Model workspace is valid after applying override inputs."
    return
  fi

  if [[ "${chosen_model}" == "Cosmos-Reason2-8B" ]]; then
    prepare_cosmos_default "${workspace_dir}" "${chosen_model}" "${edge_root}" "${edge_build}" "${plugin_path}"
  fi

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "DRY-RUN  model layout planning complete."
  else
    validate_model_layout "${workspace_dir}" "${chosen_model}" || fail \
      "Model workspace is incomplete. First-class Cosmos setup failed and optional overrides did not provide a valid layout."
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
  local isaac_ros_ws="${10}"

  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'DRY-RUN  generate %s\n' "${env_file}"
    return
  fi

  cat >"${env_file}" <<ENV
#!/usr/bin/env bash
# Generated by scripts/prepare_thor_jp72_assets.sh

export ROS_DISTRO="${ros_distro}"
export ROS_WORKSPACE="${ros_workspace}"
export TENSORRT_EDGE_LLM_ROOT="${edge_root}"
export TENSORRT_EDGE_LLM_BUILD_DIR="${edge_build}"
export EDGE_VLM_MODEL_NAME="${chosen_model}"
export EDGE_VLM_WORKSPACE_DIR="${workspace_dir}"
export EDGE_VLM_LLM_ENGINE_DIR="${llm_dir}"
export EDGE_VLM_MULTIMODAL_ENGINE_DIR="${multimodal_dir}"
export EDGELLM_PLUGIN_PATH="${plugin_path}"
export ISAAC_ROS_WS="${isaac_ros_ws}"
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
isaac_ros_ws="${ISAAC_ROS_WS:-${ros_workspace}}"
isaac_ros_ws="${isaac_ros_ws/\$\{HOME\}/${HOME}}"
isaac_ros_ws="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${isaac_ros_ws}")"
default_edge_root="$(manifest_value edge_llm.default_root)"
default_edge_build="$(manifest_value edge_llm.default_build_dir)"
default_workspace="$(manifest_value default_workspace)"
cuda_ctk_version="${EDGE_VLM_CUDA_CTK_VERSION:-$(manifest_value edge_llm.cuda_ctk_version)}"

edge_root="${TENSORRT_EDGE_LLM_ROOT:-${default_edge_root}}"
edge_root="${edge_root/\$\{HOME\}/${HOME}}"
edge_build="${TENSORRT_EDGE_LLM_BUILD_DIR:-${default_edge_build}}"
edge_build="${edge_build/\$\{HOME\}/${HOME}}"
workspace_dir="${EDGE_VLM_WORKSPACE_DIR:-${default_workspace}}"
workspace_dir="${workspace_dir/\$\{HOME\}/${HOME}}"
llm_dir="${EDGE_VLM_LLM_ENGINE_DIR:-${workspace_dir}/${model_name}/engine/llm}"
multimodal_dir="${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-${workspace_dir}/${model_name}/engine}"
plugin_path="${EDGELLM_PLUGIN_PATH:-${edge_build}/libNvInfer_edgellm_plugin.so}"

printf 'Thor JP7.2 setup plan:\n'
printf '  ROS_DISTRO: %s\n' "${ros_distro}"
printf '  ROS_WORKSPACE: %s\n' "${ros_workspace}"
printf '  ISAAC_ROS_WS: %s\n' "${isaac_ros_ws}"
printf '  Edge-LLM root: %s\n' "${edge_root}"
printf '  Edge-LLM build: %s\n' "${edge_build}"
printf '  Edge-LLM commit: %s\n' "$(manifest_value edge_llm.commit)"
printf '  Thor CUDA_CTK_VERSION: %s\n' "${cuda_ctk_version}"
printf '  Model: %s\n' "${model_name}"
printf '  Model workspace: %s\n' "${workspace_dir}"
printf '  Env file: %s\n' "${env_file}"

if [[ "${skip_edge_llm}" -eq 0 ]]; then
  prepare_edge_llm "${edge_root}" "$(manifest_value edge_llm.commit)" "${edge_build}" "${cuda_ctk_version}"
fi

if [[ "${skip_model}" -eq 0 ]]; then
  prepare_model_layout "${workspace_dir}" "${model_name}" "${edge_root}" "${edge_build}" "${plugin_path}"
fi

if [[ "${skip_rtdetr}" -eq 0 ]]; then
  install_rtdetr_models "${ros_distro}" "${isaac_ros_ws}"
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
  "${plugin_path}" \
  "${isaac_ros_ws}"

echo "Preparation completed."
