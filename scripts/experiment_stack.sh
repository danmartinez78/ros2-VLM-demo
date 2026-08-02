#!/usr/bin/env bash
# Copyright 2025 edge_vlm_ros contributors
# experiment_stack.sh — one-command lifecycle for the edge_vlm workbench stack.
#
# Usage:
#   ./scripts/experiment_stack.sh start   [options]
#   ./scripts/experiment_stack.sh stop
#   ./scripts/experiment_stack.sh status
#   ./scripts/experiment_stack.sh logs    [--server|--console] [--lines N]
#   ./scripts/experiment_stack.sh restart [options]
#
# The script manages two owned components:
#   1. edge_vlm_server  — persistent native IPC inference service
#   2. web_console      — Python workbench (python3 -m web_console)
#
# Safety guarantees:
#   - Never kills processes whose start-time does not match the stored record.
#   - Only removes sockets that are stale (no live listener) and marked as owned
#     by a previous start of this script.
#   - Rolls back all owned components after a partial startup failure.
#   - Uses SIGTERM → bounded wait → SIGKILL for graceful shutdown.
#
# Runtime state is stored in EDGE_VLM_STACK_RUN_DIR (default: ~/.edge_vlm_stack/run).
# Each component record file contains:
#   PID=<pid>
#   START_TIME=<kernel-ticks since boot from /proc/PID/stat field 22>
#   CMD=<argv[0]>
#   STARTED_AT=<iso8601>

set -Eeuo pipefail

# ── constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STACK_RUN_DIR="${EDGE_VLM_STACK_RUN_DIR:-${HOME}/.edge_vlm_stack/run}"

# Component defaults
DEFAULT_SOCKET_PATH="/tmp/edge_vlm.sock"
DEFAULT_WEB_HOST="127.0.0.1"
DEFAULT_WEB_PORT=8765
DEFAULT_RUNS_DIR="${HOME}/.web_console/runs"
# Positional args 5 and 6 for edge_vlm_server: request_timeout inference_deadline (seconds).
# The inference deadline MUST be strictly less than the request timeout.
DEFAULT_SERVER_REQUEST_TIMEOUT="${EDGE_VLM_REQUEST_TIMEOUT:-90}"
DEFAULT_SERVER_INFERENCE_DEADLINE="${EDGE_VLM_INFERENCE_DEADLINE:-60}"

# Health-check timeouts (seconds)
SERVER_AWAIT_SECONDS="${EDGE_VLM_SERVER_AWAIT:-60}"
CONSOLE_AWAIT_SECONDS="${EDGE_VLM_CONSOLE_AWAIT:-30}"
STOP_TERM_WAIT="${EDGE_VLM_STOP_TERM_WAIT:-10}"
STOP_KILL_WAIT="${EDGE_VLM_STOP_KILL_WAIT:-5}"

# Runtime file paths
SERVER_PID_FILE="${STACK_RUN_DIR}/edge_vlm_server.pid"
CONSOLE_PID_FILE="${STACK_RUN_DIR}/web_console.pid"
SERVER_LOG="${STACK_RUN_DIR}/edge_vlm_server.log"
CONSOLE_LOG="${STACK_RUN_DIR}/web_console.log"
STACK_CONFIG="${STACK_RUN_DIR}/stack.conf"
SOCKET_OWNER_FILE="${STACK_RUN_DIR}/socket.owner"

# ── usage ─────────────────────────────────────────────────────────────────────

usage() {
  cat <<'USAGE'
Usage: experiment_stack.sh <command> [options]

Commands:
  start     Load environment, validate, start edge_vlm_server and web workbench
  stop      Gracefully stop all owned stack components
  status    Print a concise health summary
  logs      Tail component logs (both by default)
  restart   Stop then start (passes options to start)

Start/restart options:
  --model MODEL       Model name or ID from the catalog to activate.
                      Looks up profile paths from model_catalog and exports them.
                      An unknown or incomplete model (missing engine/plugin dirs)
                      causes immediate startup failure.
                      When omitted, EDGE_VLM_LLM_ENGINE_DIR /
                      EDGE_VLM_MULTIMODAL_ENGINE_DIR / EDGELLM_PLUGIN_PATH must
                      already be set in the environment.
  --socket PATH       IPC socket path (default: /tmp/edge_vlm.sock)
  --host HOST         Web console bind address (default: 127.0.0.1)
  --port PORT         Web console TCP port (default: 8765)
  --runs-dir DIR      Run artifact root directory
  --server-bin PATH   Path to edge_vlm_server binary
  --cli-bin PATH      Path to edge_vlm_cli binary
  --env-file PATH     Environment file to source (default: scripts/edge_vlm_env.sh if present)
  --no-server         Skip starting edge_vlm_server (use an already-running service)
  --fg                Run in foreground (print logs to stdout, Ctrl-C to stop)
  --debug             Verbose diagnostic output

Logs options:
  --server            Show edge_vlm_server log only
  --console           Show web_console log only
  --lines N           Number of initial tail lines (default: 50)
USAGE
}

# ── helpers ───────────────────────────────────────────────────────────────────

_info()  { echo "[stack] $*"; }
_warn()  { echo "[stack] WARNING: $*" >&2; }
_error() { echo "[stack] ERROR: $*" >&2; }
_debug() { [[ "${DEBUG_STACK:-}" == "1" ]] && echo "[stack:debug] $*" >&2 || true; }

_now_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# Return field 22 (starttime in ticks) from /proc/PID/stat.
# This value is stable for the lifetime of a process and changes on PID reuse.
_proc_start_time() {
  local pid="$1"
  if [[ -r "/proc/${pid}/stat" ]]; then
    awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null || true
  fi
}

# Write a PID record file atomically via a temporary file.
# Args: <file> <pid> <cmd> <started_at>
_write_pid_file() {
  local file="$1" pid="$2" cmd="$3" started_at="$4"
  local start_time
  start_time=$(_proc_start_time "${pid}")
  local tmp="${file}.tmp.$$"
  printf 'PID=%s\nSTART_TIME=%s\nCMD=%s\nSTARTED_AT=%s\n' \
    "${pid}" "${start_time}" "${cmd}" "${started_at}" > "${tmp}"
  mv -f "${tmp}" "${file}"
}

# Read PID from record file; return empty string if file is absent or malformed.
_read_pid() {
  local file="$1"
  [[ -f "${file}" ]] || return 0
  grep '^PID=' "${file}" | cut -d= -f2 2>/dev/null || true
}

# Read stored start_time from record file.
_read_start_time() {
  local file="$1"
  [[ -f "${file}" ]] || return 0
  grep '^START_TIME=' "${file}" | cut -d= -f2 2>/dev/null || true
}

# Return the PID if the record is live and the process start-time matches.
# Returns empty string and exit 1 if the record is stale or absent.
_live_pid() {
  local file="$1"
  local stored_pid stored_st current_st
  stored_pid=$(_read_pid "${file}")
  [[ -z "${stored_pid}" ]] && return 1
  [[ ! -d "/proc/${stored_pid}" ]] && return 1
  stored_st=$(_read_start_time "${file}")
  current_st=$(_proc_start_time "${stored_pid}")
  # An empty start-time on either side means we cannot verify — treat as stale.
  if [[ -z "${stored_st}" || -z "${current_st}" ]]; then
    return 1
  fi
  [[ "${stored_st}" == "${current_st}" ]] || return 1
  echo "${stored_pid}"
}

# Stop a process referenced by a PID record file.
# Uses SIGTERM then SIGKILL after a bounded wait.
# Never sends signals unless start-time matches.
_stop_component() {
  local label="$1" pid_file="$2"
  local pid
  if ! pid=$(_live_pid "${pid_file}" 2>/dev/null); then
    local stored_pid
    stored_pid=$(_read_pid "${pid_file}" 2>/dev/null || true)
    if [[ -n "${stored_pid}" ]]; then
      _info "${label}: not running (stale PID ${stored_pid}); cleaning up record"
    else
      _info "${label}: no record; nothing to stop"
    fi
    rm -f "${pid_file}"
    return 0
  fi

  _info "${label}: sending SIGTERM to PID ${pid}"
  kill -TERM "${pid}" 2>/dev/null || true

  local deadline
  deadline=$(( $(date +%s) + STOP_TERM_WAIT ))
  while (( $(date +%s) < deadline )); do
    [[ ! -d "/proc/${pid}" ]] && break
    sleep 0.2
  done

  if [[ -d "/proc/${pid}" ]]; then
    _warn "${label}: PID ${pid} still alive after ${STOP_TERM_WAIT}s; sending SIGKILL"
    kill -KILL "${pid}" 2>/dev/null || true
    local kill_deadline
    kill_deadline=$(( $(date +%s) + STOP_KILL_WAIT ))
    while (( $(date +%s) < kill_deadline )); do
      [[ ! -d "/proc/${pid}" ]] && break
      sleep 0.2
    done
    if [[ -d "/proc/${pid}" ]]; then
      _error "${label}: PID ${pid} did not exit after SIGKILL"
    fi
  fi

  _info "${label}: stopped"
  rm -f "${pid_file}"
}

# Return 0 if the Unix domain socket at path has a live listener.
# Uses ss(8) if available, otherwise falls back to checking if the socket
# file exists (conservative: treats existing file as potentially live).
_socket_has_listener() {
  local path="$1"
  [[ -S "${path}" ]] || return 1
  if command -v ss >/dev/null 2>&1; then
    ss -lxnp 2>/dev/null | grep -qF "${path}"
  else
    # Cannot determine — treat as live to be conservative
    return 0
  fi
}

# Remove a stale socket only if it is stale (no listener) and we previously
# marked it as owned by this script (SOCKET_OWNER_FILE present).
_maybe_remove_stale_socket() {
  local path="$1"
  [[ -S "${path}" ]] || return 0
  if _socket_has_listener "${path}"; then
    return 0  # live — do not touch
  fi
  # Stale socket: only remove if we own it
  if [[ -f "${SOCKET_OWNER_FILE}" ]]; then
    local owned_path
    owned_path=$(cat "${SOCKET_OWNER_FILE}" 2>/dev/null || true)
    if [[ "${owned_path}" == "${path}" ]]; then
      _info "Removing stale owned socket: ${path}"
      rm -f "${path}"
      rm -f "${SOCKET_OWNER_FILE}"
    else
      _warn "Stale socket ${path} exists but is not ours (owned: ${owned_path}); leaving it"
    fi
  else
    _warn "Stale socket ${path} exists but is not owned by this stack; leaving it"
  fi
}

# Wait up to <seconds> for a Unix domain socket to appear with a live listener.
_await_socket() {
  local path="$1" seconds="$2" label="$3"
  local deadline
  deadline=$(( $(date +%s) + seconds ))
  while (( $(date +%s) < deadline )); do
    if _socket_has_listener "${path}"; then
      _info "${label}: socket ready"
      return 0
    fi
    sleep 1
  done
  _error "${label}: socket ${path} not ready after ${seconds}s"
  return 1
}

# Wait up to <seconds> for an HTTP endpoint to respond 200.
_await_http() {
  local url="$1" seconds="$2" label="$3"
  local deadline
  deadline=$(( $(date +%s) + seconds ))
  while (( $(date +%s) < deadline )); do
    if command -v curl >/dev/null 2>&1; then
      if curl -sf --max-time 2 "${url}" >/dev/null 2>&1; then
        _info "${label}: HTTP health check passed"
        return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -qO- --timeout=2 "${url}" >/dev/null 2>&1; then
        _info "${label}: HTTP health check passed"
        return 0
      fi
    else
      # No HTTP client; fall back to checking that the process is alive.
      return 0
    fi
    sleep 1
  done
  _error "${label}: HTTP endpoint ${url} not ready after ${seconds}s"
  return 1
}

# Check whether a TCP port is in use.
_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tlnp 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1  # cannot determine
  fi
}

# Print LAN warning when web console is bound to a non-loopback address.
_warn_if_non_loopback() {
  local host="$1"
  case "${host}" in
    127.*|::1|localhost) return 0 ;;
  esac
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  WARNING: web_console is listening on a non-loopback interface  ║"
  echo "║                                                                  ║"
  echo "║  The console has NO AUTHENTICATION.  Any client that can reach  ║"
  echo "║  this address can trigger inference and control processes.       ║"
  echo "║                                                                  ║"
  echo "║  Restrict TCP port ${host}:${WEB_PORT} to the trusted LAN subnet via ║"
  echo "║  the host firewall (e.g. ufw or iptables).                      ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
}

# ── model resolution ──────────────────────────────────────────────────────────

# Resolve a model name/ID from the catalog and export the engine/plugin paths.
# When a model name or ID is supplied, uses the Python model_catalog to look up
# the profile; then validates that all required paths exist on disk.
# Fails with a clear error message when the model is unknown or incomplete.
_resolve_model() {
  local selected_model="$1"
  if [[ -z "${selected_model}" ]]; then
    return 0
  fi

  _info "Resolving model '${selected_model}' from catalog…"
  local py_out
  if ! py_out=$(PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -c "
import sys, os
try:
    from web_console.model_catalog import discover_models
except ImportError as e:
    print(f'ImportError: {e}', file=sys.stderr)
    sys.exit(1)
target = sys.argv[1]
profiles = discover_models()
selected = next(
    (p for p in profiles if p.model_name == target or p.model_id == target),
    None,
)
if selected is None:
    known = [p.model_name for p in profiles]
    print(
        f'Model not found: {target!r}. '
        f'Known models: {known if known else \"(none discovered)\"}',
        file=sys.stderr,
    )
    sys.exit(1)
missing = []
if not selected.llm_engine_exists:
    missing.append(f'LLM engine directory: {selected.llm_engine_dir!r}')
if not selected.multimodal_engine_exists:
    missing.append(f'Multimodal engine directory: {selected.multimodal_engine_dir!r}')
if not selected.plugin_exists:
    missing.append(f'Plugin file: {selected.plugin_path!r}')
if missing:
    for m in missing:
        print(f'Missing: {m}', file=sys.stderr)
    sys.exit(1)
print(f'EDGE_VLM_LLM_ENGINE_DIR={selected.llm_engine_dir}')
print(f'EDGE_VLM_MULTIMODAL_ENGINE_DIR={selected.multimodal_engine_dir}')
print(f'EDGELLM_PLUGIN_PATH={selected.plugin_path}')
print(f'EDGE_VLM_MODEL_NAME={selected.model_name}')
print(f'EDGE_VLM_MODEL_ID={selected.model_id}')
" "${selected_model}" 2>&1); then
    _error "Model resolution failed: ${py_out}"
    return 1
  fi

  # Export each KEY=VALUE line; split on the first '=' to preserve path values.
  local key value
  while IFS= read -r line; do
    key="${line%%=*}"
    value="${line#*=}"
    [[ -n "${key}" ]] && export "${key}=${value}"
  done <<< "${py_out}"

  _info "Model resolved: ${EDGE_VLM_MODEL_NAME:-${selected_model}}"
  _debug "  LLM engine    : ${EDGE_VLM_LLM_ENGINE_DIR:-}"
  _debug "  Multimodal    : ${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}"
  _debug "  Plugin        : ${EDGELLM_PLUGIN_PATH:-}"
}

# ── validation ────────────────────────────────────────────────────────────────

_validate() {
  local ok=0

  # Validate Python environment
  if ! command -v python3 >/dev/null 2>&1; then
    _error "python3 not found on PATH"
    ok=1
  elif ! python3 -c "import web_console" 2>/dev/null; then
    # Try from repo root
    if ! PYTHONPATH="${REPO_ROOT}" python3 -c "import web_console" 2>/dev/null; then
      _warn "web_console package not importable from current PYTHONPATH; will try REPO_ROOT"
    fi
  fi

  # Validate service binary (skip if --no-server)
  if [[ "${NO_SERVER:-0}" != "1" ]]; then
    local server_bin="${SERVER_BIN}"
    if ! command -v "${server_bin}" >/dev/null 2>&1 && [[ ! -x "${server_bin}" ]]; then
      _warn "edge_vlm_server binary not found at '${server_bin}'"
      _warn "Set SERVER_BIN or pass --server-bin, or use --no-server to skip"
      _warn "Proceeding without native inference service (workbench only)"
      NO_SERVER=1
    fi
  fi

  # Validate CLI binary
  if ! command -v "${CLI_BIN}" >/dev/null 2>&1 && [[ ! -x "${CLI_BIN}" ]]; then
    _warn "edge_vlm_cli binary not found at '${CLI_BIN}'"
    _warn "Standalone inference via web console will fail until the CLI is available"
  fi

  # Validate required engine bundle when starting the server.
  # Missing or nonexistent engine/plugin configuration is a hard startup failure;
  # the server cannot initialise without these positional arguments.
  if [[ "${NO_SERVER:-0}" != "1" ]]; then
    local missing_cfg=0
    if [[ -z "${EDGE_VLM_LLM_ENGINE_DIR:-}" ]]; then
      _error "EDGE_VLM_LLM_ENGINE_DIR is not set. Set it via env/env-file, or use --model <name>."
      missing_cfg=1
    elif [[ ! -d "${EDGE_VLM_LLM_ENGINE_DIR}" ]]; then
      _error "EDGE_VLM_LLM_ENGINE_DIR does not exist: ${EDGE_VLM_LLM_ENGINE_DIR}"
      missing_cfg=1
    fi
    if [[ -z "${EDGE_VLM_MULTIMODAL_ENGINE_DIR:-}" ]]; then
      _error "EDGE_VLM_MULTIMODAL_ENGINE_DIR is not set. Set it via env/env-file, or use --model <name>."
      missing_cfg=1
    elif [[ ! -d "${EDGE_VLM_MULTIMODAL_ENGINE_DIR}" ]]; then
      _error "EDGE_VLM_MULTIMODAL_ENGINE_DIR does not exist: ${EDGE_VLM_MULTIMODAL_ENGINE_DIR}"
      missing_cfg=1
    fi
    if [[ -z "${EDGELLM_PLUGIN_PATH:-}" ]]; then
      _error "EDGELLM_PLUGIN_PATH is not set. Set it via env/env-file, or use --model <name>."
      missing_cfg=1
    elif [[ ! -f "${EDGELLM_PLUGIN_PATH}" ]]; then
      _error "EDGELLM_PLUGIN_PATH does not exist: ${EDGELLM_PLUGIN_PATH}"
      missing_cfg=1
    fi
    if [[ "${missing_cfg}" == "1" ]]; then
      _error "Engine/plugin configuration is incomplete. Use --no-server to skip the inference service."
      ok=1
    fi
  fi

  # Validate output directories
  local runs_dir="${RUNS_DIR}"
  if [[ -n "${runs_dir}" ]] && [[ ! -d "${runs_dir}" ]]; then
    mkdir -p "${runs_dir}" || { _error "Cannot create runs directory: ${runs_dir}"; ok=1; }
  fi

  # Check for port conflict
  if _port_in_use "${WEB_PORT}"; then
    # Check if it's our own console
    local our_pid
    our_pid=$(_live_pid "${CONSOLE_PID_FILE}" 2>/dev/null || true)
    if [[ -z "${our_pid}" ]]; then
      _error "Port ${WEB_PORT} is already in use by an unrelated process"
      _info "Run 'ss -tlnp | grep :${WEB_PORT}' for details"
      ok=1
    fi
  fi

  return "${ok}"
}

# ── commands ──────────────────────────────────────────────────────────────────

cmd_start() {
  mkdir -p "${STACK_RUN_DIR}"

  # ── check for already-running stack ──────────────────────────────────────

  local server_pid console_pid
  server_pid=$(_live_pid "${SERVER_PID_FILE}" 2>/dev/null || true)
  console_pid=$(_live_pid "${CONSOLE_PID_FILE}" 2>/dev/null || true)

  if [[ -n "${server_pid}" || -n "${console_pid}" ]]; then
    _info "Stack components already running:"
    [[ -n "${server_pid}" ]] && _info "  edge_vlm_server: PID ${server_pid}"
    [[ -n "${console_pid}" ]] && _info "  web_console:     PID ${console_pid}"
    _info "Use 'restart' to stop and start fresh, or 'status' for details."
    return 0
  fi

  # ── load environment ──────────────────────────────────────────────────────

  local env_file="${ENV_FILE:-}"
  if [[ -z "${env_file}" ]]; then
    local default_env="${SCRIPT_DIR}/edge_vlm_env.sh"
    [[ -f "${default_env}" ]] && env_file="${default_env}"
  fi
  if [[ -n "${env_file}" ]]; then
    if [[ -f "${env_file}" ]]; then
      _info "Loading environment from ${env_file}"
      # shellcheck source=/dev/null
      source "${env_file}"
    else
      _error "Environment file not found: ${env_file}"
      return 1
    fi
  fi

  # ── resolve model (if --model was given) ─────────────────────────────────

  _resolve_model "${SELECTED_MODEL:-}" || return 1

  # ── validate ──────────────────────────────────────────────────────────────

  _warn_if_non_loopback "${WEB_HOST}"
  _validate || return 1

  # ── write config snapshot ─────────────────────────────────────────────────

  cat > "${STACK_CONFIG}" <<CONF
SOCKET_PATH=${SOCKET_PATH}
WEB_HOST=${WEB_HOST}
WEB_PORT=${WEB_PORT}
RUNS_DIR=${RUNS_DIR}
SERVER_BIN=${SERVER_BIN}
CLI_BIN=${CLI_BIN}
NO_SERVER=${NO_SERVER:-0}
ACTIVE_MODEL=${EDGE_VLM_MODEL_NAME:-}
ACTIVE_MODEL_LLM_DIR=${EDGE_VLM_LLM_ENGINE_DIR:-}
STARTED_AT=$(_now_iso)
CONF

  # ── cleanup stale socket ─────────────────────────────────────────────────

  _maybe_remove_stale_socket "${SOCKET_PATH}"

  # ── start edge_vlm_server ─────────────────────────────────────────────────

  local server_started=0
  if [[ "${NO_SERVER:-0}" != "1" ]]; then
    # Build the full argv array for edge_vlm_server.  The binary takes six
    # positional arguments (no flags); order matches run_standalone_service_smoke.sh:
    #   1. LLM engine directory
    #   2. Multimodal engine directory
    #   3. EdgeLLM TensorRT plugin path
    #   4. IPC socket path
    #   5. Worker request timeout (seconds)
    #   6. Worker inference deadline (seconds, must be < timeout)
    local server_argv=(
      "${SERVER_BIN}"
      "${EDGE_VLM_LLM_ENGINE_DIR}"
      "${EDGE_VLM_MULTIMODAL_ENGINE_DIR}"
      "${EDGELLM_PLUGIN_PATH}"
      "${SOCKET_PATH}"
      "${SERVER_REQUEST_TIMEOUT}"
      "${SERVER_INFERENCE_DEADLINE}"
    )
    _info "Starting edge_vlm_server → log: ${SERVER_LOG}"
    _debug "  argv: ${server_argv[*]}"
    if [[ "${FOREGROUND:-0}" == "1" ]]; then
      "${server_argv[@]}" >> "${SERVER_LOG}" 2>&1 &
    else
      "${server_argv[@]}" > "${SERVER_LOG}" 2>&1 &
    fi
    local srv_pid=$!
    _write_pid_file "${SERVER_PID_FILE}" "${srv_pid}" "${SERVER_BIN}" "$(_now_iso)"
    # Mark socket ownership
    echo "${SOCKET_PATH}" > "${SOCKET_OWNER_FILE}"
    server_started=1

    _info "Waiting for IPC socket ${SOCKET_PATH} (up to ${SERVER_AWAIT_SECONDS}s)…"
    if ! _await_socket "${SOCKET_PATH}" "${SERVER_AWAIT_SECONDS}" "edge_vlm_server"; then
      _error "edge_vlm_server did not create IPC socket in time"
      _info "Rolling back: stopping edge_vlm_server"
      _stop_component "edge_vlm_server (rollback)" "${SERVER_PID_FILE}"
      return 1
    fi
  else
    _info "Skipping edge_vlm_server (--no-server)"
  fi

  # ── start web_console ────────────────────────────────────────────────────

  local console_args=(
    python3 -m web_console
    --host "${WEB_HOST}"
    --port "${WEB_PORT}"
    --socket "${SOCKET_PATH}"
    --cli "${CLI_BIN}"
    --runs-dir "${RUNS_DIR}"
  )

  _info "Starting web_console → log: ${CONSOLE_LOG}"
  if [[ "${FOREGROUND:-0}" == "1" ]]; then
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
      "${console_args[@]}" >> "${CONSOLE_LOG}" 2>&1 &
  else
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
      "${console_args[@]}" > "${CONSOLE_LOG}" 2>&1 &
  fi
  local con_pid=$!
  _write_pid_file "${CONSOLE_PID_FILE}" "${con_pid}" "python3" "$(_now_iso)"

  _info "Waiting for HTTP on ${WEB_HOST}:${WEB_PORT} (up to ${CONSOLE_AWAIT_SECONDS}s)…"
  if ! _await_http "http://${WEB_HOST}:${WEB_PORT}/api/status" "${CONSOLE_AWAIT_SECONDS}" "web_console"; then
    _error "web_console did not respond in time"
    _info "Rolling back: stopping web_console"
    _stop_component "web_console (rollback)" "${CONSOLE_PID_FILE}"
    if [[ "${server_started}" == "1" ]]; then
      _info "Rolling back: stopping edge_vlm_server"
      _stop_component "edge_vlm_server (rollback)" "${SERVER_PID_FILE}"
    fi
    return 1
  fi

  _info ""
  _info "Stack started successfully."
  [[ "${server_started}" == "1" ]] && _info "  edge_vlm_server : PID $(_read_pid "${SERVER_PID_FILE}")"
  _info "  web_console     : PID $(_read_pid "${CONSOLE_PID_FILE}")"
  _info "  Web workbench   : http://${WEB_HOST}:${WEB_PORT}/"
  _info "  IPC socket      : ${SOCKET_PATH}"
  _info "  Logs            : ${STACK_RUN_DIR}"

  if [[ "${FOREGROUND:-0}" == "1" ]]; then
    _info "Running in foreground; press Ctrl-C to stop."
    # Tail both logs to stdout
    tail -f "${SERVER_LOG}" "${CONSOLE_LOG}" &
    local tail_pid=$!
    trap "cmd_stop; kill ${tail_pid} 2>/dev/null || true" INT TERM
    wait "${con_pid}" || true
    cmd_stop
    kill "${tail_pid}" 2>/dev/null || true
  fi
}

cmd_stop() {
  local found=0

  local server_pid console_pid
  server_pid=$(_live_pid "${SERVER_PID_FILE}" 2>/dev/null || true)
  console_pid=$(_live_pid "${CONSOLE_PID_FILE}" 2>/dev/null || true)

  if [[ -n "${console_pid}" ]]; then
    found=1
    _stop_component "web_console" "${CONSOLE_PID_FILE}"
  else
    [[ -f "${CONSOLE_PID_FILE}" ]] && { _info "web_console: stale record removed"; rm -f "${CONSOLE_PID_FILE}"; }
  fi

  if [[ -n "${server_pid}" ]]; then
    found=1
    _stop_component "edge_vlm_server" "${SERVER_PID_FILE}"
    # After stopping the server, the socket file is now stale — safe to remove
    if [[ -f "${SOCKET_OWNER_FILE}" ]]; then
      local owned_path
      owned_path=$(cat "${SOCKET_OWNER_FILE}" 2>/dev/null || true)
      if [[ -S "${owned_path}" ]]; then
        _info "Removing owned socket: ${owned_path}"
        rm -f "${owned_path}"
      fi
      rm -f "${SOCKET_OWNER_FILE}"
    fi
  else
    [[ -f "${SERVER_PID_FILE}" ]] && { _info "edge_vlm_server: stale record removed"; rm -f "${SERVER_PID_FILE}"; }
  fi

  if [[ "${found}" == "0" ]]; then
    _info "No owned stack components were running."
  else
    _info "Stack stopped."
  fi
}

cmd_status() {
  local overall_ok=1

  # ── component status ─────────────────────────────────────────────────────

  local server_pid console_pid
  server_pid=$(_live_pid "${SERVER_PID_FILE}" 2>/dev/null || true)
  console_pid=$(_live_pid "${CONSOLE_PID_FILE}" 2>/dev/null || true)

  if [[ -n "${server_pid}" ]]; then
    echo "  edge_vlm_server : running (PID ${server_pid})"
  elif [[ -f "${SERVER_PID_FILE}" ]]; then
    echo "  edge_vlm_server : stopped (stale PID $(_read_pid "${SERVER_PID_FILE}"))"
    overall_ok=0
  else
    echo "  edge_vlm_server : stopped (no record)"
  fi

  if [[ -n "${console_pid}" ]]; then
    echo "  web_console     : running (PID ${console_pid})"
  elif [[ -f "${CONSOLE_PID_FILE}" ]]; then
    echo "  web_console     : stopped (stale PID $(_read_pid "${CONSOLE_PID_FILE}"))"
    overall_ok=0
  else
    echo "  web_console     : stopped (no record)"
  fi

  # ── IPC socket ───────────────────────────────────────────────────────────

  local sock_path="${SOCKET_PATH:-/tmp/edge_vlm.sock}"
  if [[ -f "${STACK_CONFIG}" ]]; then
    local cfg_sock
    cfg_sock=$(grep '^SOCKET_PATH=' "${STACK_CONFIG}" | cut -d= -f2 2>/dev/null || true)
    [[ -n "${cfg_sock}" ]] && sock_path="${cfg_sock}"
  fi

  if _socket_has_listener "${sock_path}" 2>/dev/null; then
    echo "  IPC socket      : reachable (${sock_path})"
  elif [[ -S "${sock_path}" ]]; then
    echo "  IPC socket      : stale (${sock_path})"
    overall_ok=0
  else
    echo "  IPC socket      : absent (${sock_path})"
  fi

  # ── HTTP health ──────────────────────────────────────────────────────────

  local web_host="${WEB_HOST:-127.0.0.1}"
  local web_port="${WEB_PORT:-8765}"
  if [[ -f "${STACK_CONFIG}" ]]; then
    local cfg_host cfg_port
    cfg_host=$(grep '^WEB_HOST=' "${STACK_CONFIG}" | cut -d= -f2 2>/dev/null || true)
    cfg_port=$(grep '^WEB_PORT=' "${STACK_CONFIG}" | cut -d= -f2 2>/dev/null || true)
    [[ -n "${cfg_host}" ]] && web_host="${cfg_host}"
    [[ -n "${cfg_port}" ]] && web_port="${cfg_port}"
  fi
  local url="http://${web_host}:${web_port}/api/status"

  if command -v curl >/dev/null 2>&1; then
    if curl -sf --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "  web workbench   : healthy (${url})"
    else
      echo "  web workbench   : not responding (${url})"
      overall_ok=0
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -qO- --timeout=2 "${url}" >/dev/null 2>&1; then
      echo "  web workbench   : healthy (${url})"
    else
      echo "  web workbench   : not responding (${url})"
      overall_ok=0
    fi
  else
    echo "  web workbench   : unknown (curl/wget not available for health check)"
  fi

  echo ""
  if [[ "${overall_ok}" == "1" && -n "${console_pid}" ]]; then
    echo "  status: OK"
  else
    echo "  status: degraded (see above)"
  fi
  # ── active model ──────────────────────────────────────────────────────────
  if [[ -f "${STACK_CONFIG}" ]]; then
    local cfg_model cfg_llm_dir
    cfg_model=$(grep '^ACTIVE_MODEL=' "${STACK_CONFIG}" | cut -d= -f2 2>/dev/null || true)
    cfg_llm_dir=$(grep '^ACTIVE_MODEL_LLM_DIR=' "${STACK_CONFIG}" | cut -d= -f2 2>/dev/null || true)
    if [[ -n "${cfg_model}" ]]; then
      echo "  active model    : ${cfg_model}"
      [[ -n "${cfg_llm_dir}" ]] && echo "  llm engine dir  : ${cfg_llm_dir}"
    fi
  fi
  echo "  runtime dir: ${STACK_RUN_DIR}"
}

cmd_logs() {
  local show_server=0 show_console=0 lines=50
  while (($#)); do
    case "$1" in
      --server)  show_server=1; shift ;;
      --console) show_console=1; shift ;;
      --lines)   lines="$2"; shift 2 ;;
      --lines=*) lines="${1#--lines=}"; shift ;;
      *)
        _error "Unknown logs option: $1"
        usage >&2
        return 2
        ;;
    esac
  done

  # Default: show both
  if [[ "${show_server}" == "0" && "${show_console}" == "0" ]]; then
    show_server=1
    show_console=1
  fi

  local log_files=()
  if [[ "${show_server}" == "1" && -f "${SERVER_LOG}" ]]; then
    log_files+=("${SERVER_LOG}")
  elif [[ "${show_server}" == "1" ]]; then
    _warn "edge_vlm_server log not found: ${SERVER_LOG}"
  fi
  if [[ "${show_console}" == "1" && -f "${CONSOLE_LOG}" ]]; then
    log_files+=("${CONSOLE_LOG}")
  elif [[ "${show_console}" == "1" ]]; then
    _warn "web_console log not found: ${CONSOLE_LOG}"
  fi

  if [[ "${#log_files[@]}" == "0" ]]; then
    _error "No log files found in ${STACK_RUN_DIR}"
    return 1
  fi

  tail -f -n "${lines}" "${log_files[@]}"
}

# ── argument parsing ──────────────────────────────────────────────────────────

COMMAND="${1:-help}"
shift || true

SOCKET_PATH="${DEFAULT_SOCKET_PATH}"
WEB_HOST="${DEFAULT_WEB_HOST}"
WEB_PORT="${DEFAULT_WEB_PORT}"
RUNS_DIR="${DEFAULT_RUNS_DIR}"
SERVER_BIN="${EDGE_VLM_SERVER_BIN:-edge_vlm_server}"
CLI_BIN="${EDGE_VLM_CLI_BIN:-edge_vlm_cli}"
SERVER_REQUEST_TIMEOUT="${DEFAULT_SERVER_REQUEST_TIMEOUT}"
SERVER_INFERENCE_DEADLINE="${DEFAULT_SERVER_INFERENCE_DEADLINE}"
ENV_FILE=""
NO_SERVER=0
FOREGROUND=0
SELECTED_MODEL=""

# Temporary array for logs sub-command options
LOGS_OPTS=()

case "${COMMAND}" in
  start|restart|status|stop) ;;
  logs)
    LOGS_OPTS=("$@")
    set --
    ;;
  help|--help|-h)
    usage
    exit 0
    ;;
  *)
    _error "Unknown command: ${COMMAND}"
    usage >&2
    exit 2
    ;;
esac

while (($#)); do
  case "$1" in
    --model)       SELECTED_MODEL="$2"; shift 2 ;;
    --model=*)     SELECTED_MODEL="${1#--model=}"; shift ;;
    --socket)      SOCKET_PATH="$2"; shift 2 ;;
    --socket=*)    SOCKET_PATH="${1#--socket=}"; shift ;;
    --host)        WEB_HOST="$2"; shift 2 ;;
    --host=*)      WEB_HOST="${1#--host=}"; shift ;;
    --port)        WEB_PORT="$2"; shift 2 ;;
    --port=*)      WEB_PORT="${1#--port=}"; shift ;;
    --runs-dir)    RUNS_DIR="$2"; shift 2 ;;
    --runs-dir=*)  RUNS_DIR="${1#--runs-dir=}"; shift ;;
    --server-bin)  SERVER_BIN="$2"; shift 2 ;;
    --server-bin=*)SERVER_BIN="${1#--server-bin=}"; shift ;;
    --cli-bin)     CLI_BIN="$2"; shift 2 ;;
    --cli-bin=*)   CLI_BIN="${1#--cli-bin=}"; shift ;;
    --env-file)    ENV_FILE="$2"; shift 2 ;;
    --env-file=*)  ENV_FILE="${1#--env-file=}"; shift ;;
    --no-server)   NO_SERVER=1; shift ;;
    --fg|--foreground) FOREGROUND=1; shift ;;
    --debug)       DEBUG_STACK=1; shift ;;
    --help|-h)     usage; exit 0 ;;
    *)
      _error "Unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

export DEBUG_STACK="${DEBUG_STACK:-0}"

# ── dispatch ──────────────────────────────────────────────────────────────────

case "${COMMAND}" in
  start)
    cmd_start
    ;;
  stop)
    cmd_stop
    ;;
  status)
    cmd_status
    ;;
  logs)
    cmd_logs "${LOGS_OPTS[@]+"${LOGS_OPTS[@]}"}"
    ;;
  restart)
    _info "Stopping existing stack…"
    cmd_stop
    _info "Starting stack…"
    cmd_start
    ;;
esac
