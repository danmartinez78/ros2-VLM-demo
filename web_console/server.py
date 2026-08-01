# Copyright 2025 edge_vlm_ros contributors
"""
HTTP server for the local web experiment console.

Binds to 127.0.0.1 by default. Passing ``--host 0.0.0.0`` (or any
non-loopback address) exposes an unauthenticated process-control API on
that interface; a conspicuous warning is printed at startup in that case.

Routes
------
GET  /                        → HTML console page
GET  /api/status              → JSON server + GPU status
POST /api/infer               → multipart/form-data image + prompt → JSON result
POST /api/ros/start           → JSON body → start ROS experiment
POST /api/ros/stop            → JSON body {run_id} → stop ROS experiment
GET  /api/runs                → JSON list of recent runs
GET  /api/runs/<run_id>       → JSON single run manifest
GET  /api/runs/<run_id>/logs  → JSON log lines for a ROS run
GET  /static/<path>           → static asset (css/js)
"""
from __future__ import annotations

import datetime
import email.parser
import email.policy
import html
import json
import os
import pathlib
import re
import shutil
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from .inference_client import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    run_inference,
)
from .process_manager import ProcessManager
from .run_store import RunStore, _is_safe_run_id, _TERMINAL_STATUSES
from .status_collector import collect_status

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# Allowlisted ROS experiment parameters (no arbitrary paths or shell expansion).
_ALLOWED_ROS_PARAMS = frozenset(
    {
        "image_topic",
        "prompt",
        "observation_history_max_entries",
        "observation_history_max_chars",
        "max_generate_length",
        "success_results_required",
        "playback_duration",
        "result_timeout",
        "instruction_delivery_mode",
    }
)

_STATIC_DIR = pathlib.Path(__file__).parent / "static"
_CONTENT_TYPES: Dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ico": "image/x-icon",
}

# Pre-load the static file allowlist at import time so that file paths used in
# I/O operations are never derived from user-provided request paths.
# This is also the fix for CodeQL py/path-injection: the values in the map are
# resolved from the known static directory, not from any user input.
def _build_static_map() -> Dict[str, pathlib.Path]:
    mapping: Dict[str, pathlib.Path] = {}
    try:
        for entry in _STATIC_DIR.iterdir():
            if entry.is_file():
                mapping[entry.name] = entry.resolve()
    except OSError:
        pass
    return mapping


_STATIC_MAP: Dict[str, pathlib.Path] = _build_static_map()


# ── helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _json_response(data: Any) -> bytes:
    return json.dumps(data, indent=2).encode("utf-8")


def _parse_multipart(content_type: str, body: bytes) -> Dict[str, Any]:
    """Parse multipart/form-data body. Returns dict of field-name → value."""
    parser = email.parser.BytesParser(policy=email.policy.default)
    msg = parser.parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    parts: Dict[str, Any] = {}
    for part in msg.iter_parts():
        params = dict(part.get_params(header="Content-Disposition") or [])
        name = params.get("name", "")
        if not name:
            continue
        filename = params.get("filename", None)
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            parts[name] = {"filename": filename, "data": payload}
        else:
            parts[name] = payload.decode(errors="replace")
    return parts


def _validate_ros_params(params: Dict[str, Any]) -> Optional[str]:
    """Return an error string if any parameter is invalid, else None."""
    for key in params:
        if key not in _ALLOWED_ROS_PARAMS:
            return f"Unknown ROS parameter: {key!r}"
    # Validate instruction_delivery_mode
    mode = params.get("instruction_delivery_mode", "inline")
    if mode not in ("inline", "structured"):
        return "instruction_delivery_mode must be 'inline' or 'structured'"
    # Validate integer fields
    int_fields = {
        "observation_history_max_entries": (0, 256),
        "observation_history_max_chars": (0, 1_000_000),
        "max_generate_length": (1, 4096),
        "success_results_required": (1, 100),
        "playback_duration": (1, 3600),
        "result_timeout": (1, 3600),
    }
    for field, (lo, hi) in int_fields.items():
        if field not in params:
            continue
        try:
            val = int(params[field])
        except (TypeError, ValueError):
            return f"{field} must be an integer"
        if not lo <= val <= hi:
            return f"{field} must be in [{lo}, {hi}]"
    return None


# ── request handler ───────────────────────────────────────────────────────────

class ConsoleHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the web experiment console."""

    server_instance: "ConsoleServer"  # set by ConsoleServer

    # ── routing ───────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/":
                self._serve_index()
            elif path == "/api/status":
                self._api_get_status()
            elif path == "/api/runs":
                self._api_list_runs()
            elif _RUN_RE.match(path) and path.endswith("/logs"):
                run_id = _RUN_RE.match(path).group(1)
                self._api_get_run_logs(run_id)
            elif _RUN_RE.match(path):
                run_id = _RUN_RE.match(path).group(1)
                self._api_get_run(run_id)
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            else:
                self._send_error(404, "Not found")
        except Exception:
            self.server_instance.log_error(traceback.format_exc())
            self._send_error(500, "Internal server error")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/api/infer":
                self._api_infer()
            elif path == "/api/ros/start":
                self._api_ros_start()
            elif path == "/api/ros/stop":
                self._api_ros_stop()
            else:
                self._send_error(404, "Not found")
        except Exception:
            self.server_instance.log_error(traceback.format_exc())
            self._send_error(500, "Internal server error")

    # ── static + HTML ─────────────────────────────────────────────────────────

    def _serve_index(self) -> None:
        body = _render_index(self.server_instance)
        self._send_response(200, body, "text/html; charset=utf-8")

    def _serve_static(self, rel_path: str) -> None:
        # Use the pre-computed static-file map so the path used for I/O is
        # never derived from user input — this avoids py/path-injection.
        # Only the bare filename is matched; no subdirectory traversal is allowed.
        filename = pathlib.PurePosixPath(rel_path).name
        static_path = _STATIC_MAP.get(filename)
        if static_path is None:
            self._send_error(404, "Static file not found")
            return
        ext = static_path.suffix.lower()
        ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
        self._send_response(200, static_path.read_bytes(), ctype)

    # ── API ───────────────────────────────────────────────────────────────────

    def _api_get_status(self) -> None:
        cfg = self.server_instance.config
        status = collect_status(cfg.get("socket_path", ""))
        status["active_ros_run_id"] = self.server_instance.process_manager.active_ros_run_id()
        self._send_json(200, status)

    def _api_list_runs(self) -> None:
        runs = self.server_instance.run_store.list_runs()
        self._send_json(200, {"runs": runs})

    def _api_get_run(self, run_id: str) -> None:
        if not _is_safe_run_id(run_id):
            self._send_error(400, "Invalid run_id")
            return
        record = self.server_instance.run_store.get_run(run_id)
        if record is None:
            self._send_error(404, "Run not found")
            return
        self._send_json(200, record)

    def _api_get_run_logs(self, run_id: str) -> None:
        if not _is_safe_run_id(run_id):
            self._send_error(400, "Invalid run_id")
            return
        # In-memory logs (available while the process is still registered).
        live_logs = self.server_instance.process_manager.get_logs(run_id)
        # Persisted manifest — authoritative after finalization or after restart.
        record = self.server_instance.run_store.get_run(run_id)
        status = record.get("status") if record else None
        terminal = status in _TERMINAL_STATUSES
        # Prefer live logs while running; fall back to persisted logs after restart.
        log_lines = live_logs if live_logs else (record.get("log_lines", []) if record else [])
        self._send_json(
            200,
            {"run_id": run_id, "log_lines": log_lines, "status": status, "terminal": terminal},
        )

    def _api_infer(self) -> None:
        cfg = self.server_instance.config
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_error(400, "Expected multipart/form-data")
            return

        length_str = self.headers.get("Content-Length", "")
        try:
            length = int(length_str)
        except (ValueError, TypeError):
            self._send_error(411, "Content-Length required")
            return
        if length < 0:
            self._send_error(400, "Invalid Content-Length")
            return
        if length > MAX_IMAGE_BYTES + 65536:  # image + form fields overhead
            self._send_error(413, "Upload too large")
            return

        body = self.rfile.read(length)
        try:
            parts = _parse_multipart(content_type, body)
        except Exception as exc:
            self._send_error(400, f"Multipart parse error: {exc}")
            return

        prompt = parts.get("prompt", "").strip()
        if not prompt:
            self._send_error(400, "prompt field is required")
            return

        image_part = parts.get("image")
        if not isinstance(image_part, dict):
            self._send_error(400, "image file field is required")
            return

        filename = image_part.get("filename", "upload")
        ext = pathlib.Path(filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            self._send_error(
                415,
                f"Unsupported image type {ext!r}. "
                f"Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
            )
            return

        image_data: bytes = image_part["data"]
        if len(image_data) > MAX_IMAGE_BYTES:
            self._send_error(413, "Image too large")
            return
        if len(image_data) == 0:
            self._send_error(400, "Empty image upload")
            return

        # Validate numeric parameters from form.
        try:
            max_gen = int(parts.get("max_generate_length", "64"))
            temperature = float(parts.get("temperature", "0.2"))
            top_p = float(parts.get("top_p", "0.9"))
            top_k = int(parts.get("top_k", "20"))
        except (TypeError, ValueError) as exc:
            self._send_error(400, f"Invalid parameter: {exc}")
            return

        run_id = RunStore.new_run_id()
        tmp_dir = tempfile.mkdtemp(prefix="web_console_infer_")
        try:
            image_path = os.path.join(tmp_dir, f"upload{ext}")
            with open(image_path, "wb") as fh:
                fh.write(image_data)

            cli_path = cfg.get("cli_path", "edge_vlm_cli")
            socket_path = cfg.get("socket_path", "")
            timeout = cfg.get("infer_timeout_seconds", 120)

            result = run_inference(
                cli_path=cli_path,
                socket_path=socket_path,
                image_path=image_path,
                prompt=prompt,
                max_generate_length=max_gen,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                timeout_seconds=timeout,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        record: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "kind": "standalone",
            "created_at": _now_iso(),
            "prompt": prompt,
            "image_filename": filename,
            "max_generate_length": max_gen,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "success": result.success,
            "text": result.text,
            "error": result.error,
            "inference_seconds": round(result.inference_seconds, 3),
        }
        self.server_instance.run_store.save_run(run_id, record)
        self._send_json(200 if result.success else 422, record)

    def _api_ros_start(self) -> None:
        cfg = self.server_instance.config
        body = self._read_json_body()
        if body is None:
            return

        params = body.get("params", {})
        error = _validate_ros_params(params)
        if error:
            self._send_error(400, error)
            return

        run_id = RunStore.new_run_id()
        run_store = self.server_instance.run_store

        # Create the per-run artifact directory (same as the run store dir for this run).
        artifact_dir = run_store.base_dir / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Only allowlisted env vars are forwarded; no arbitrary env injection.
        env = _build_ros_env(params, cfg, artifact_dir)

        script_path = cfg.get(
            "ros_script_path",
            str(
                pathlib.Path(__file__).parent.parent
                / "scripts"
                / "test_data"
                / "run_image_proc_test.sh"
            ),
        )

        # Construct argument list; never shell=True.
        args = ["bash", script_path]

        # ── persist initial manifest BEFORE launching ────────────────────────
        # Writing the manifest here (with status "starting") guarantees that
        # the completion callback always finds a record even when the child
        # exits before start_ros_experiment returns (fast-exit race).
        initial_record: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "kind": "ros",
            "created_at": _now_iso(),
            "params": params,
            "status": "starting",
            "artifact_dir": str(artifact_dir),
        }
        run_store.save_run(run_id, initial_record)

        def _on_complete(
            rid: str,
            exit_code: Optional[int],
            was_stopped: bool,
            log_lines: list,
        ) -> None:
            """Atomically finalize the manifest with terminal state + artifacts."""
            if was_stopped:
                status_str: str = "stopped"
                success: bool = False
            elif exit_code == 0:
                status_str = "completed"
                success = True
            else:
                status_str = "failed"
                success = False

            # Collect JSON artifact files written by the ROS script into the
            # per-run directory (e.g. result.json, metrics.json).
            run_dir = run_store.run_dir(rid)
            artifacts: Dict[str, Any] = {}
            if run_dir is not None:
                for candidate in ("result.json", "metrics.json"):
                    p = run_dir / candidate
                    if p.is_file():
                        try:
                            artifacts[candidate] = json.loads(
                                p.read_text(encoding="utf-8")
                            )
                        except (json.JSONDecodeError, OSError):
                            artifacts[candidate] = None

            updates: Dict[str, Any] = {
                "status": status_str,
                "exit_code": exit_code,
                "completed_at": _now_iso(),
                "success": success,
                "log_lines": log_lines[:200],
            }
            if artifacts:
                updates["artifacts"] = artifacts

            # finalize_run is a no-op if the run is already terminal (idempotency guard).
            run_store.finalize_run(rid, updates)

        try:
            pid = self.server_instance.process_manager.start_ros_experiment(
                run_id=run_id,
                args=args,
                env=env,
                on_complete=_on_complete,
            )
        except RuntimeError as exc:
            # Another ROS experiment is already running; mark the pre-written
            # manifest as failed so it is not left in a non-terminal state.
            run_store.finalize_run(run_id, {
                "status": "failed",
                "exit_code": None,
                "completed_at": _now_iso(),
                "success": False,
                "error": str(exc),
            })
            self._send_error(409, str(exc))
            return
        except Exception as exc:
            # Popen or pgid lookup failed; finalize the manifest as failed.
            run_store.finalize_run(run_id, {
                "status": "failed",
                "exit_code": None,
                "completed_at": _now_iso(),
                "success": False,
                "error": str(exc),
            })
            self._send_error(500, f"Failed to launch ROS experiment: {exc}")
            return

        # Atomically advance "starting" → "running" only if the completion
        # callback has not already written a terminal state (fast-exit guard).
        run_store.update_run_if_status(run_id, "starting", {"status": "running", "pid": pid})
        resp_record = run_store.get_run(run_id) or initial_record
        self._send_json(202, resp_record)

    def _api_ros_stop(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        run_id = body.get("run_id", "")
        if not _is_safe_run_id(run_id):
            self._send_error(400, "Invalid or missing run_id")
            return
        try:
            self.server_instance.process_manager.stop_experiment(run_id)
        except KeyError:
            self._send_error(404, f"No active experiment with run_id={run_id!r}")
            return
        self._send_json(200, {"run_id": run_id, "status": "stopped"})

    # ── response helpers ──────────────────────────────────────────────────────

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        length_str = self.headers.get("Content-Length", "")
        try:
            length = int(length_str)
        except (ValueError, TypeError):
            self._send_error(411, "Content-Length required")
            return None
        if length < 0:
            self._send_error(400, "Invalid Content-Length")
            return None
        if length > 65536:
            self._send_error(413, "Request body too large")
            return None
        body = self.rfile.read(length)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return None

    def _send_json(self, status: int, data: Any) -> None:
        body = _json_response(data)
        self._send_response(status, body, "application/json; charset=utf-8")

    def _send_error(self, status: int, message: str) -> None:
        body = _json_response({"error": message})
        self._send_response(status, body, "application/json; charset=utf-8")

    def _send_response(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        # Forward to server logger to allow suppression in tests.
        self.server_instance.log_request(fmt % args)


# ── routing regex ─────────────────────────────────────────────────────────────

_RUN_RE = re.compile(
    r"^/api/runs/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:/logs)?$"
)


# ── ROS env builder ───────────────────────────────────────────────────────────

def _build_ros_env(
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    artifact_dir: Optional[pathlib.Path] = None,
) -> Dict[str, str]:
    """Build a subprocess environment for the ROS experiment script.

    Only allowlisted parameters are passed; no arbitrary values are forwarded.
    When *artifact_dir* is supplied it is set as ARTIFACT_DIR so the script
    knows where to write result/metrics files.
    """
    env = os.environ.copy()
    _map: Dict[str, str] = {
        "image_topic": "IMAGE_TOPIC",
        "prompt": "TEST_PROMPT",
        "observation_history_max_entries": "OBSERVATION_HISTORY_MAX_ENTRIES",
        "observation_history_max_chars": "OBSERVATION_HISTORY_MAX_CHARS",
        "max_generate_length": "MAX_GENERATE_LENGTH",
        "success_results_required": "SUCCESS_RESULTS_REQUIRED",
        "playback_duration": "PLAYBACK_DURATION_SECONDS",
        "result_timeout": "RESULT_TIMEOUT_SECONDS",
        "instruction_delivery_mode": "INSTRUCTION_DELIVERY_MODE",
    }
    for param_key, env_key in _map.items():
        if param_key in params:
            env[env_key] = str(params[param_key])
    # Set ARTIFACT_DIR from the explicit per-run path when provided, otherwise
    # fall back to the legacy cfg key (kept for backward compatibility).
    if artifact_dir is not None:
        env["ARTIFACT_DIR"] = str(artifact_dir)
    elif cfg.get("artifact_dir"):
        env["ARTIFACT_DIR"] = str(cfg["artifact_dir"])
    return env


# ── server class ──────────────────────────────────────────────────────────────

class ConsoleServer(ThreadingHTTPServer):
    """Threading HTTP server for the web console with shared state.

    Uses ThreadingHTTPServer so that long-running inference requests do not
    block concurrent status polls or stop commands from the browser.  All
    shared state (ProcessManager, RunStore) is independently thread-safe.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        config: Optional[Dict[str, Any]] = None,
        process_manager: Optional[ProcessManager] = None,
        run_store: Optional[RunStore] = None,
    ) -> None:
        self.config = config or {}
        self.process_manager = process_manager or ProcessManager()
        runs_dir = pathlib.Path(
            self.config.get("runs_dir", pathlib.Path.home() / ".web_console" / "runs")
        )
        self.run_store = run_store or RunStore(runs_dir)
        self._quiet = self.config.get("quiet", False)

        def handler(*args: Any, **kwargs: Any) -> None:
            h = ConsoleHandler(*args, **kwargs)
            h.server_instance = self
            return h  # type: ignore[return-value]

        # Use a closure so handler.__init__ sets server_instance before serving.
        class _Handler(ConsoleHandler):
            server_instance = self  # class-level reference

        super().__init__((host, port), _Handler)

    def log_request(self, message: str) -> None:
        if not self._quiet:
            print(f"[web_console] {message}", flush=True)

    def log_error(self, message: str) -> None:
        print(f"[web_console ERROR] {message}", flush=True)

    def shutdown_gracefully(self) -> None:
        self.process_manager.cleanup()
        self.shutdown()


# ── HTML rendering ────────────────────────────────────────────────────────────

def _render_index(srv: ConsoleServer) -> bytes:
    cfg = srv.config
    socket_path = html.escape(cfg.get("socket_path", "/tmp/edge_vlm.sock"))
    runs = srv.run_store.list_runs()
    runs_html = _render_runs_table(runs)
    active_run = srv.process_manager.active_ros_run_id() or ""
    page = _INDEX_TEMPLATE.replace("{{SOCKET_PATH}}", socket_path)
    page = page.replace("{{RUNS_TABLE}}", runs_html)
    page = page.replace("{{ACTIVE_ROS_RUN}}", html.escape(active_run))
    return page.encode("utf-8")


def _render_runs_table(runs: list) -> str:
    if not runs:
        return "<p class='muted'>No runs yet.</p>"
    rows = []
    for r in runs:
        run_id = html.escape(r.get("run_id", ""))
        kind = html.escape(r.get("kind", ""))
        created = html.escape(r.get("created_at", ""))
        success = r.get("success", None)
        badge = ""
        if success is True:
            badge = "<span class='badge ok'>OK</span>"
        elif success is False:
            badge = "<span class='badge fail'>FAIL</span>"
        rows.append(
            f"<tr>"
            f"<td><a href='#' onclick=\"loadRun('{run_id}')\">{run_id[:8]}…</a></td>"
            f"<td>{kind}</td>"
            f"<td>{created}</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )
    return (
        "<table class='runs'><thead><tr>"
        "<th>Run</th><th>Kind</th><th>Created</th><th>Status</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


# ── HTML template ─────────────────────────────────────────────────────────────

_INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>edge_vlm_ros — Web Console</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header>
  <h1>edge_vlm_ros Web Console</h1>
  <span id="server-badge" class="badge">checking…</span>
</header>
<main>

<section id="status-section">
  <h2>Status</h2>
  <pre id="status-out">Loading…</pre>
  <button onclick="refreshStatus()">Refresh</button>
</section>

<section id="infer-section">
  <h2>Standalone Inference</h2>
  <form id="infer-form" enctype="multipart/form-data">
    <label>Image (max 64 MiB): <input type="file" name="image" accept="image/*" required></label><br>
    <label>Prompt: <input type="text" name="prompt" value="Describe the scene." size="60" required></label><br>
    <label>Max tokens: <input type="number" name="max_generate_length" value="64" min="1" max="4096"></label>
    <label>Temperature: <input type="number" name="temperature" value="0.2" step="0.01" min="0" max="2"></label>
    <label>top-p: <input type="number" name="top_p" value="0.9" step="0.01" min="0.01" max="1"></label>
    <label>top-k: <input type="number" name="top_k" value="20" min="1" max="200"></label><br>
    <button type="button" onclick="submitInfer()">Run Inference</button>
  </form>
  <pre id="infer-out"></pre>
</section>

<section id="ros-section">
  <h2>ROS Experiment</h2>
  <div id="ros-active" class="active-run-banner" style="display:none">
    Active run: <span id="ros-active-id">{{ACTIVE_ROS_RUN}}</span>
    <button onclick="stopRos()">Stop</button>
  </div>
  <div id="ros-form-area">
    <label>Image topic: <input id="ros-topic" type="text" value="/hawk_0_left_rgb_image" size="40"></label><br>
    <label>Prompt: <input id="ros-prompt" type="text" value="" placeholder="(optional override)" size="60"></label><br>
    <label>Max tokens: <input id="ros-max-gen" type="number" value="64" min="1" max="4096"></label>
    <label>Delivery mode:
      <select id="ros-delivery">
        <option value="inline">inline</option>
        <option value="structured">structured</option>
      </select>
    </label>
    <label>Obs. history entries: <input id="ros-hist-entries" type="number" value="0" min="0" max="256"></label>
    <label>Obs. history chars: <input id="ros-hist-chars" type="number" value="0" min="0" max="1000000"></label>
    <label>Playback duration (s): <input id="ros-playback" type="number" value="20" min="1" max="3600"></label>
    <label>Result timeout (s): <input id="ros-timeout" type="number" value="120" min="1" max="3600"></label>
    <label>Required results: <input id="ros-required" type="number" value="1" min="1" max="100"></label><br>
    <button type="button" onclick="startRos()">Start ROS Experiment</button>
  </div>
  <pre id="ros-out"></pre>
  <div id="ros-logs-area" style="display:none">
    <h3>Live Logs</h3>
    <pre id="ros-logs"></pre>
    <button onclick="pollLogs()">Refresh Logs</button>
  </div>
</section>

<section id="history-section">
  <h2>Run History</h2>
  <button onclick="refreshHistory()">Refresh</button>
  <div id="runs-table">{{RUNS_TABLE}}</div>
  <div id="run-detail" style="display:none">
    <h3>Run Detail</h3>
    <pre id="run-detail-out"></pre>
  </div>
</section>

</main>
<script src="/static/app.js"></script>
<script>
  refreshStatus();
  var activeRos = "{{ACTIVE_ROS_RUN}}";
  if (activeRos) { showActiveRun(activeRos); }
</script>
</body>
</html>
"""
