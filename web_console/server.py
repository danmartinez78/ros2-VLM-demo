# Copyright 2025 edge_vlm_ros contributors
"""
HTTP server for the local web experiment console.

Binds to 127.0.0.1 by default. Passing ``--host 0.0.0.0`` (or any
non-loopback address) exposes an unauthenticated process-control API on
that interface; a conspicuous warning is printed at startup in that case.

Routes
------
GET  /                                          → HTML console page
GET  /api/status                                → JSON server + GPU status
POST /api/infer                                 → multipart/form-data image + prompt → JSON result
POST /api/ros/start                             → JSON body → start ROS experiment
POST /api/ros/stop                              → JSON body {run_id} → stop ROS experiment
GET  /api/runs                                  → JSON list of recent runs
GET  /api/runs/<run_id>                         → JSON single run manifest
GET  /api/runs/<run_id>/logs                    → JSON log lines for a ROS run
GET  /static/<path>                             → static asset (css/js)
GET  /api/profiles                              → JSON list of task profiles
GET  /api/frame-datasets                        → JSON list of frame datasets
GET  /api/frame-datasets/<dataset_id>           → JSON frame dataset manifest
GET  /api/frame-datasets/<dataset_id>/frames/<n> → JPEG frame image (bounded, allowlisted)
POST /api/extract                               → JSON body → start frame extraction from catalog bag
POST /api/extract/<run_id>/cancel               → cancel frame extraction run
GET  /api/runs/<run_id>/reviews                 → JSON list of review annotations for a run
POST /api/runs/<run_id>/reviews                 → JSON body → upsert review annotation
GET  /api/compare                               → JSON comparison of two runs aligned by frame
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

from .dataset_catalog import build_download_command, discover_datasets
from .experiment_engine import (
    ExperimentDefinition,
    _VALID_STRATEGIES,
    run_experiment,
    validate_definition,
)
from .frame_extractor import (
    ExtractionParams,
    FrameDatasetStore,
    _is_safe_dataset_id,
    allowlist_bag_path,
    build_extraction_args,
    validate_extraction_params,
)
from .inference_client import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    run_inference,
)
from .model_catalog import discover_models
from .process_manager import ProcessManager
from .review_store import (
    ALLOWED_REVIEW_LABELS,
    ReviewAnnotation,
    ReviewStore,
    validate_review,
)
from .run_store import RunStore, _is_safe_run_id, _TERMINAL_STATUSES
from .status_collector import collect_status, check_server_reachable
from .task_profiles import (
    ParsedOutput,
    TaskProfile,
    discover_profiles,
    parse_structured_output,
)

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
        "rosbag_path",
    }
)

# Allowlisted artifact filenames produced by run_image_proc_test.sh.
# The script writes these into $ARTIFACT_DIR; no other filenames are captured.
_ROS_ARTIFACT_ALLOWLIST = ("manifest.json", "benchmark.jsonl", "launch.log", "results.log")

# Maximum frame index that can be requested via the image-serving route.
_MAX_FRAME_INDEX = 9999

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


def _single_quote_close_idx(s: str) -> int:
    """Return the index of the first unescaped closing single-quote in *s*, or -1.

    Within a YAML single-quoted scalar ``''`` is an escaped literal quote and
    does **not** close the scalar; a lone ``'`` does.
    """
    i = 0
    while i < len(s):
        if s[i] == "'":
            if i + 1 < len(s) and s[i + 1] == "'":
                i += 2  # escaped '' — skip both chars
            else:
                return i  # unescaped ' — closing quote
        else:
            i += 1
    return -1


def _fold_single_quoted_scalar(parts: list) -> str:
    """Apply YAML single-quoted scalar line-folding to a list of string parts.

    Rules (matching YAML 1.2 §7.4 single-quoted scalars):
      - Adjacent non-empty parts are joined with a single space.
      - Empty parts (blank physical lines) introduce a ``\\n`` in the result.
      - ``''`` sequences are decoded to a literal ``'``.
    """
    segments: list = []
    pending_nl = 0
    for part in parts:
        if not part:
            pending_nl += 1
        else:
            if segments:
                if pending_nl:
                    segments.append("\n" * pending_nl)
                else:
                    segments.append(" ")
            segments.append(part)
            pending_nl = 0
    return "".join(segments).replace("''", "'")


def _parse_results_log(text: str) -> list:
    """Parse real ros2 topic echo output for VlmResult messages.

    Real ``ros2 topic echo`` output starts each message with the message fields
    and ends each message with a ``---`` separator.  The parser accumulates
    top-level scalar fields until the trailing ``---`` marker flushes the
    collected values as a completed frame.  Incomplete trailing content (no
    closing ``---``) is discarded.

    Nested fields under ``header:`` → ``stamp:`` are parsed to extract
    ``sec`` and ``nanosec`` and returned as ``source_timestamp_ns``.

    Block-scalar indicators (``>`` ``>-`` ``|`` ``|-``) are handled by
    collecting subsequent indented lines as a single space-joined string.

    Single-quoted scalars that span multiple physical lines (as emitted by
    ``ros2 topic echo`` for long ``response`` fields) are reassembled using
    YAML single-quoted line-folding: blank lines become ``\\n``, and adjacent
    non-empty continuation lines are joined with a space.

    Field names are normalised to the canonical UI schema on output:
      * ``frame_sequence`` → ``frame_seq``
      * ``response``       → ``text``
      * ``inference_seconds`` → ``latency_ms``  (multiplied by 1 000)

    Returns an empty list on empty or entirely malformed input.
    """
    frames: list = []
    current: Dict[str, Any] = {}
    pending_key: Optional[str] = None
    pending_lines: list = []
    # Single-quoted multi-line scalar state
    pending_sq_key: Optional[str] = None
    pending_sq_lines: list = []
    _stamp_sec: Optional[int] = None
    _stamp_nanosec: Optional[int] = None
    _in_header: bool = False
    _in_stamp: bool = False

    def _flush_frame() -> None:
        nonlocal current, pending_key, pending_lines
        nonlocal pending_sq_key, pending_sq_lines
        nonlocal _stamp_sec, _stamp_nanosec, _in_header, _in_stamp
        # Finalise any pending block-scalar value.
        if pending_key is not None:
            current[pending_key] = " ".join(l.strip() for l in pending_lines)
            pending_key = None
            pending_lines = []
        # Finalise any pending single-quoted multi-line scalar.
        if pending_sq_key is not None:
            current[pending_sq_key] = _fold_single_quoted_scalar(pending_sq_lines)
            pending_sq_key = None
            pending_sq_lines = []
        if not current:
            # Reset timestamp state even for empty blocks.
            _stamp_sec = None
            _stamp_nanosec = None
            _in_header = False
            _in_stamp = False
            return
        # Normalise to the canonical UI schema.
        frame: Dict[str, Any] = {}
        for k, v in current.items():
            if k == "frame_sequence":
                frame["frame_seq"] = v
            elif k == "response":
                frame["text"] = v
            elif k == "inference_seconds":
                try:
                    frame["latency_ms"] = float(v) * 1000.0
                except (TypeError, ValueError):
                    frame["latency_ms"] = v
            else:
                frame[k] = v
        # Attach source timestamp when both components were parsed.
        if _stamp_sec is not None:
            ns = _stamp_sec * 1_000_000_000
            if _stamp_nanosec is not None:
                ns += _stamp_nanosec
            frame["source_timestamp_ns"] = ns
        frames.append(frame)
        current = {}
        _stamp_sec = None
        _stamp_nanosec = None
        _in_header = False
        _in_stamp = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "---":
            _flush_frame()
            continue
        # ── single-quoted multi-line scalar continuation ──────────────────────
        if pending_sq_key is not None:
            close_idx = _single_quote_close_idx(stripped)
            if close_idx >= 0:
                # Found the closing quote on this line.
                pending_sq_lines.append(stripped[:close_idx])
                current[pending_sq_key] = _fold_single_quoted_scalar(pending_sq_lines)
                pending_sq_key = None
                pending_sq_lines = []
            else:
                pending_sq_lines.append(stripped)
            continue
        # ── block-scalar continuation (indented under the pending key) ────────
        if pending_key is not None:
            if line.startswith((" ", "\t")):
                pending_lines.append(stripped)
                continue
            else:
                current[pending_key] = " ".join(l.strip() for l in pending_lines)
                pending_key = None
                pending_lines = []
        # Indented lines: track header → stamp → sec/nanosec hierarchy.
        if line.startswith((" ", "\t")):
            if _in_header and stripped == "stamp:":
                _in_stamp = True
            elif _in_stamp and ": " in stripped:
                k, _, v = stripped.partition(": ")
                k = k.strip()
                v = v.strip()
                try:
                    if k == "sec":
                        _stamp_sec = int(v)
                    elif k == "nanosec":
                        _stamp_nanosec = int(v)
                except ValueError:
                    pass
            continue
        # Top-level line: reset nesting context.
        _in_stamp = False
        if stripped == "header:":
            _in_header = True
            continue
        _in_header = False
        # Top-level key: value pair.
        if ": " in line:
            key, _, raw = line.partition(": ")
            key = key.strip()
            raw = raw.strip()
            if raw in (">", ">-", "|", "|-"):
                pending_key = key
                pending_lines = []
            elif raw.startswith("'"):
                # Single-quoted scalar: check if it closes on the same line.
                inner = raw[1:]  # strip the opening '
                close_idx = _single_quote_close_idx(inner)
                if close_idx >= 0:
                    # Entire scalar is on this line.
                    val_str = inner[:close_idx].replace("''", "'")
                    if val_str.lower() == "true":
                        current[key] = True
                    elif val_str.lower() == "false":
                        current[key] = False
                    elif val_str == "":
                        current[key] = ""
                    else:
                        try:
                            current[key] = (
                                float(val_str)
                                if ("." in val_str or "e" in val_str.lower())
                                else int(val_str)
                            )
                        except ValueError:
                            current[key] = val_str
                else:
                    # Multi-line single-quoted scalar: opening content on this line.
                    pending_sq_key = key
                    pending_sq_lines = [inner]
            else:
                val_str = raw.strip("\"")
                if val_str.lower() == "true":
                    current[key] = True
                elif val_str.lower() == "false":
                    current[key] = False
                elif val_str == "":
                    current[key] = ""
                else:
                    try:
                        current[key] = (
                            float(val_str) if ("." in val_str or "e" in val_str.lower()) else int(val_str)
                        )
                    except ValueError:
                        current[key] = val_str
    # Incomplete trailing content (no closing ---) is intentionally discarded.
    return frames

def _parse_benchmark_jsonl(text: str) -> list:
    """Parse a benchmark.jsonl file; returns all valid JSON object records.

    Malformed lines are silently skipped so a single corrupt entry does not
    discard an entire session.
    """
    records: list = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
        except json.JSONDecodeError:
            pass
    return records


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
            elif path == "/api/models":
                self._api_list_models()
            elif path == "/api/datasets":
                self._api_list_datasets()
            elif path == "/api/profiles":
                self._api_list_profiles()
            elif path == "/api/frame-datasets":
                self._api_list_frame_datasets()
            elif path == "/api/compare":
                self._api_compare_runs(parsed)
            elif _RUN_RE.match(path) and path.endswith("/logs"):
                run_id = _RUN_RE.match(path).group(1)
                self._api_get_run_logs(run_id)
            elif _RUN_REVIEWS_RE.match(path):
                run_id = _RUN_REVIEWS_RE.match(path).group(1)
                self._api_get_reviews(run_id)
            elif _RUN_RE.match(path):
                run_id = _RUN_RE.match(path).group(1)
                self._api_get_run(run_id)
            elif _FRAME_IMAGE_RE.match(path):
                m = _FRAME_IMAGE_RE.match(path)
                self._api_serve_frame_image(m.group(1), int(m.group(2)))
            elif _FRAME_DATASET_RE.match(path):
                dataset_id = _FRAME_DATASET_RE.match(path).group(1)
                self._api_get_frame_dataset(dataset_id)
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
            elif path == "/api/experiment/run":
                self._api_experiment_run()
            elif path == "/api/datasets/download":
                self._api_dataset_download()
            elif path == "/api/extract":
                self._api_extract_start()
            elif _EXTRACT_CANCEL_RE.match(path):
                run_id = _EXTRACT_CANCEL_RE.match(path).group(1)
                self._api_extract_cancel(run_id)
            elif _RUN_REVIEWS_RE.match(path):
                run_id = _RUN_REVIEWS_RE.match(path).group(1)
                self._api_upsert_review(run_id)
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

        # Validate rosbag_path against the installed catalog (allowlist).
        # Arbitrary filesystem paths are never accepted.
        rosbag_path = params.get("rosbag_path", "")
        if rosbag_path:
            catalog = discover_datasets(
                rosbag_root=cfg.get("rosbag_dir"),
                image_root=cfg.get("image_dataset_dir"),
                video_root=cfg.get("video_dataset_dir"),
            )
            allowed_paths = {
                b["local_path"]
                for b in catalog["rosbags"]
                if b.get("installed") and b.get("local_path")
            }
            if rosbag_path not in allowed_paths:
                self._send_error(
                    400, "rosbag_path is not an installed catalog path"
                )
                return

        run_id = RunStore.new_run_id()
        run_store = self.server_instance.run_store

        # Use a nested artifacts/ subdirectory for ARTIFACT_DIR so that the
        # script's own manifest.json (and other outputs) never overwrite the
        # console-owned manifest at <run_id>/manifest.json.
        artifact_dir = run_store.base_dir / run_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Detect whether the standalone inference service is already reachable.
        # If so, run the script in external-service mode (START_WORKER=false)
        # so the ROS adapter reuses the running service instead of starting
        # a competing worker. The service's socket and process are left untouched.
        socket_path = cfg.get("socket_path", "")
        server_status = check_server_reachable(socket_path) if socket_path else {"reachable": False}
        use_external_worker: bool = bool(server_status.get("reachable"))

        # Only allowlisted env vars are forwarded; no arbitrary env injection.
        env = _build_ros_env(params, cfg, artifact_dir, start_worker=not use_external_worker)

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
            "external_worker": use_external_worker,
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

            # Scan the nested artifacts/ subdirectory for the bounded set of
            # files written by run_image_proc_test.sh.  Log files are referenced
            # by relative path only (not embedded); manifest.json is parsed and
            # stored as script_manifest so the console manifest is never
            # overwritten by the script's own manifest.json.
            run_dir = run_store.run_dir(rid)
            artifact_rel_paths: list = []
            script_manifest: Optional[Dict[str, Any]] = None
            result_frames: list = []
            benchmark_summary: Optional[Dict[str, Any]] = None
            if run_dir is not None:
                artifacts_subdir = run_dir / "artifacts"
                if artifacts_subdir.is_dir():
                    for name in _ROS_ARTIFACT_ALLOWLIST:
                        p = artifacts_subdir / name
                        if p.is_file():
                            artifact_rel_paths.append(f"artifacts/{name}")
                            if name == "manifest.json":
                                try:
                                    script_manifest = json.loads(
                                        p.read_text(encoding="utf-8")
                                    )
                                except (json.JSONDecodeError, OSError):
                                    pass
                    # Parse structured results for display
                    results_path = artifacts_subdir / "results.log"
                    bench_path = artifacts_subdir / "benchmark.jsonl"
                    if results_path.is_file():
                        try:
                            result_frames = _parse_results_log(
                                results_path.read_text(encoding="utf-8", errors="replace")
                            )[:50]
                        except OSError:
                            pass
                    if bench_path.is_file():
                        try:
                            bench_records = _parse_benchmark_jsonl(
                                bench_path.read_text(encoding="utf-8", errors="replace")
                            )
                            frame_recs = [
                                r for r in bench_records
                                if r.get("record_type") == "frame"
                            ]
                            session_end = next(
                                (
                                    r for r in reversed(bench_records)
                                    if r.get("record_type") == "session_end"
                                ),
                                {},
                            )
                            if frame_recs:
                                success_recs = [
                                    f for f in frame_recs if f.get("success", True)
                                ]
                                infer_ms_vals = []
                                for frame in success_recs:
                                    if "inference_seconds" in frame:
                                        infer_ms_vals.append(
                                            float(frame["inference_seconds"]) * 1000.0
                                        )
                                    elif "inference_ms" in frame:
                                        infer_ms_vals.append(
                                            float(frame["inference_ms"])
                                        )
                                benchmark_summary = {
                                    "frame_count": len(frame_recs),
                                    "successful_frames": len(success_recs),
                                    "failed_frames": len(frame_recs) - len(success_recs),
                                    "dropped_frames": session_end.get(
                                        "dropped",
                                        sum(
                                            int(f.get("dropped_before", 0))
                                            for f in frame_recs
                                        ),
                                    ),
                                    "mean_inference_ms": (
                                        round(
                                            sum(infer_ms_vals) / len(infer_ms_vals),
                                            2,
                                        )
                                        if infer_ms_vals else None
                                    ),
                                    "min_inference_ms": (
                                        round(min(infer_ms_vals), 2)
                                        if infer_ms_vals else None
                                    ),
                                    "max_inference_ms": (
                                        round(max(infer_ms_vals), 2)
                                        if infer_ms_vals else None
                                    ),
                                   # Source label for UI clarity.
                                   "source": "benchmark.jsonl (all processed inference samples)",
                                }
                                # When benchmark and ROS-topic counts diverge, add
                                # an explanatory note so the UI can surface it.
                                if len(result_frames) != len(frame_recs):
                                   benchmark_summary["count_note"] = (
                                       f"benchmark.jsonl recorded {len(frame_recs)} "
                                       f"processed inference sample(s); "
                                       f"results.log captured {len(result_frames)} "
                                       f"frame result(s) from the ROS topic subscriber. "
                                       f"The difference typically reflects frames whose "
                                       f"inference completed during graceful shutdown after "
                                       f"the subscriber had already closed."
                                   )
                        except OSError:
                            pass

            updates: Dict[str, Any] = {
                "status": status_str,
                "exit_code": exit_code,
                "completed_at": _now_iso(),
                "success": success,
                "log_lines": log_lines[:200],
            }
            if artifact_rel_paths:
                updates["artifacts"] = artifact_rel_paths
            if script_manifest is not None:
                updates["script_manifest"] = script_manifest
            if result_frames:
                updates["result_frames"] = result_frames
            if benchmark_summary is not None:
                updates["benchmark_summary"] = benchmark_summary

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

    # ── model and dataset catalog API ─────────────────────────────────────────

    def _api_list_models(self) -> None:
        cfg = self.server_instance.config
        workspace_dir = cfg.get("workspace_dir", None)
        profiles = discover_models(workspace_dir=workspace_dir)
        self._send_json(200, {"models": [p.to_dict() for p in profiles]})

    def _api_list_datasets(self) -> None:
        cfg = self.server_instance.config
        catalog = discover_datasets(
            rosbag_root=cfg.get("rosbag_dir"),
            image_root=cfg.get("image_dataset_dir"),
            video_root=cfg.get("video_dataset_dir"),
        )
        self._send_json(200, catalog)

    def _api_dataset_download(self) -> None:
        """Initiate a rosbag download via the existing download_rosbags.sh script.

        Accepts ``{"bag_key": "<key>"}`` in the JSON body.  Only keys registered
        in the dataset catalog are accepted.  The command is constructed as an
        argument array — never shell=True.
        """
        cfg = self.server_instance.config
        body = self._read_json_body()
        if body is None:
            return

        bag_key = body.get("bag_key", "")
        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", bag_key):
            self._send_error(400, "Invalid bag_key")
            return

        import pathlib as _pathlib
        script_path = str(
            _pathlib.Path(__file__).parent.parent
            / "scripts"
            / "test_data"
            / "download_rosbags.sh"
        )
        args = build_download_command(script_path, bag_key)
        if args is None:
            self._send_error(400, f"Unknown or non-downloadable bag_key: {bag_key!r}")
            return

        run_id = RunStore.new_run_id()
        run_store = self.server_instance.run_store

        artifact_dir = run_store.base_dir / run_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if cfg.get("rosbag_dir"):
            env["ROSBAG_DIR"] = cfg["rosbag_dir"]

        initial_record: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "kind": "download",
            "created_at": _now_iso(),
            "bag_key": bag_key,
            "status": "starting",
        }
        run_store.save_run(run_id, initial_record)

        def _on_complete(
            rid: str,
            exit_code: Optional[int],
            was_stopped: bool,
            log_lines: list,
        ) -> None:
            status_str = "stopped" if was_stopped else ("completed" if exit_code == 0 else "failed")
            run_store.finalize_run(rid, {
                "status": status_str,
                "exit_code": exit_code,
                "completed_at": _now_iso(),
                "success": exit_code == 0 and not was_stopped,
                "log_lines": log_lines[:200],
            })

        try:
            pid = self.server_instance.process_manager.start_ros_experiment(
                run_id=run_id,
                args=args,
                env=env,
                on_complete=_on_complete,
            )
        except Exception as exc:
            run_store.finalize_run(run_id, {
                "status": "failed",
                "exit_code": None,
                "completed_at": _now_iso(),
                "success": False,
                "error": str(exc),
            })
            self._send_error(500, f"Failed to start download: {exc}")
            return

        run_store.update_run_if_status(run_id, "starting", {"status": "running", "pid": pid})
        self._send_json(202, run_store.get_run(run_id) or initial_record)

    # ── ROS-independent experiment API ────────────────────────────────────────

    def _api_experiment_run(self) -> None:
        """Submit a ROS-independent experiment run.

        Accepts a JSON body with keys:
          strategy               — "single_frame" or "single_frame_observation_history"
          image_paths            — list of absolute image paths (max 10 000)
          task_prompt            — task prompt string
          system_instruction     — optional system instruction
          observation_history_max_entries  — int (default 0)
          observation_history_max_chars    — int (default 4000)
          max_generate_length    — int (default 96)
          temperature            — float (default 0.2)
          top_p                  — float (default 0.9)
          top_k                  — int (default 20)
          timeout_seconds        — int (default 120)
          notes                  — optional reproducibility notes

        Image paths must be absolute and the files must exist at request time.
        The experiment runs in a background thread to avoid blocking the HTTP
        server.
        """
        cfg = self.server_instance.config
        body = self._read_json_body()
        if body is None:
            return

        # ── build ExperimentDefinition ───────────────────────────────────────
        try:
            defn = ExperimentDefinition(
                strategy=str(body.get("strategy", "single_frame")),
                image_paths=[str(p) for p in body.get("image_paths", [])],
                task_prompt=str(body.get("task_prompt", "")).strip(),
                system_instruction=str(
                    body.get(
                        "system_instruction",
                        "You are a vision observer. Base claims on the current image.",
                    )
                ),
                observation_history_max_entries=int(
                    body.get("observation_history_max_entries", 0)
                ),
                observation_history_max_chars=int(
                    body.get("observation_history_max_chars", 4000)
                ),
                max_generate_length=int(body.get("max_generate_length", 96)),
                temperature=float(body.get("temperature", 0.2)),
                top_p=float(body.get("top_p", 0.9)),
                top_k=int(body.get("top_k", 20)),
                timeout_seconds=int(body.get("timeout_seconds", 120)),
                notes=str(body.get("notes", "")),
            )
        except (TypeError, ValueError) as exc:
            self._send_error(400, f"Invalid parameter: {exc}")
            return

        error = validate_definition(defn)
        if error:
            self._send_error(400, error)
            return

        # Validate that ALL image files exist before marking the run active.
        # A partial check (e.g. first-100-only) allows large runs to start with
        # unchecked paths that will fail mid-experiment.
        for img_path in defn.image_paths:
            if not os.path.isabs(img_path):
                self._send_error(400, f"image_paths must be absolute: {img_path!r}")
                return
            if not os.path.isfile(img_path):
                self._send_error(400, f"Image file not found: {img_path!r}")
                return

        run_id = defn.experiment_id
        defn.run_id = run_id
        srv = self.server_instance
        run_store = srv.run_store

        # Bounded experiment coordinator: reject concurrent submissions.
        with srv._active_experiment_lock:
            if srv._active_experiment_id is not None:
                self._send_error(
                    409,
                    f"Experiment already in progress: {srv._active_experiment_id}. "
                    "Wait for it to complete or poll /api/runs/<id> for status.",
                )
                return
            srv._active_experiment_id = run_id

        artifact_dir = run_store.base_dir / run_id / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        initial_record: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "kind": "experiment",
            "strategy": defn.strategy,
            "created_at": defn.created_at,
            "image_count": len(defn.image_paths),
            "task_prompt": defn.task_prompt,
            "observation_history_max_entries": defn.observation_history_max_entries,
            "max_generate_length": defn.max_generate_length,
            "temperature": defn.temperature,
            "top_p": defn.top_p,
            "top_k": defn.top_k,
            "timeout_seconds": defn.timeout_seconds,
            "notes": defn.notes,
            "status": "running",
        }
        run_store.save_run(run_id, initial_record)

        cli_path = cfg.get("cli_path", "edge_vlm_cli")
        socket_path = cfg.get("socket_path", "/tmp/edge_vlm.sock")

        def _run_in_background() -> None:
            try:
                results = run_experiment(
                    defn,
                    cli_path=cli_path,
                    socket_path=socket_path,
                    artifact_dir=artifact_dir,
                )
                successful = [r for r in results if r.success]
                failed = [r for r in results if not r.success]
                latencies = [r.latency_ms for r in successful]
                run_store.finalize_run(run_id, {
                    "status": "completed",
                    "completed_at": _now_iso(),
                    "success": True,
                    "successful_frames": len(successful),
                    "failed_frames": len(failed),
                    "repetition_flags": sum(1 for r in results if r.repetition_flag),
                    "mean_latency_ms": (
                        round(sum(latencies) / len(latencies), 2)
                        if latencies else None
                    ),
                    "result_frames": [
                        {
                            "frame_index": r.frame_index,
                            "success": r.success,
                            "text": r.text,
                            "error": r.error,
                            "latency_ms": r.latency_ms,
                            "history_entries_used": r.history_entries_used,
                            "repetition_flag": r.repetition_flag,
                        }
                        for r in results[:50]  # cap at 50 for manifest size
                    ],
                })
            except Exception as exc:
                run_store.finalize_run(run_id, {
                    "status": "failed",
                    "completed_at": _now_iso(),
                    "success": False,
                    "error": str(exc),
                })
            finally:
                # Release the coordinator slot so the next experiment can start.
                with srv._active_experiment_lock:
                    if srv._active_experiment_id == run_id:
                        srv._active_experiment_id = None

        threading.Thread(target=_run_in_background, daemon=True).start()
        self._send_json(202, initial_record)

    # ── task profile API ─────────────────────────────────────────────────────

    def _api_list_profiles(self) -> None:
        cfg = self.server_instance.config
        profiles_dir = cfg.get("task_profiles_dir") or str(
            pathlib.Path(__file__).parent.parent / "config" / "task_profiles"
        )
        try:
            profiles = discover_profiles(profiles_dir)
        except Exception as exc:
            self._send_error(500, f"Failed to load profiles: {exc}")
            return
        self._send_json(200, {
            "profiles": [
                {
                    "profile_id": p.profile_id(),
                    "name": p.name,
                    "version": p.version,
                    "prompt_hash": p.prompt_hash,
                    "system_instruction": p.system_instruction,
                    "task_prompt": p.task_prompt,
                    "schema_example": p.schema_example,
                }
                for p in profiles
            ]
        })

    # ── frame dataset API ────────────────────────────────────────────────────

    def _api_list_frame_datasets(self) -> None:
        store = self.server_instance.frame_dataset_store
        try:
            datasets = store.list_datasets()
        except Exception as exc:
            self._send_error(500, f"Failed to list frame datasets: {exc}")
            return
        self._send_json(200, {"datasets": datasets})

    def _api_get_frame_dataset(self, dataset_id: str) -> None:
        if not _is_safe_dataset_id(dataset_id):
            self._send_error(400, "Invalid dataset_id")
            return
        store = self.server_instance.frame_dataset_store
        manifest = store.get_manifest(dataset_id)
        if manifest is None:
            self._send_error(404, "Dataset not found")
            return
        self._send_json(200, manifest)

    def _api_serve_frame_image(self, dataset_id: str, frame_index: int) -> None:
        if not _is_safe_dataset_id(dataset_id):
            self._send_error(400, "Invalid dataset_id")
            return
        if frame_index < 0 or frame_index > _MAX_FRAME_INDEX:
            self._send_error(400, "Frame index out of range")
            return
        store = self.server_instance.frame_dataset_store
        img_path = store.get_frame_path(dataset_id, frame_index)
        if img_path is None:
            self._send_error(404, "Frame not found")
            return
        try:
            data = img_path.read_bytes()
        except OSError:
            self._send_error(404, "Frame not found")
            return
        self._send_response(200, data, "image/jpeg")

    # ── frame extraction API ─────────────────────────────────────────────────

    def _api_extract_start(self) -> None:
        """Start a rosbag frame-extraction run.

        The bag must be an installed catalog entry; no arbitrary paths.
        """
        cfg = self.server_instance.config
        body = self._read_json_body()
        if body is None:
            return

        # Validate extraction parameters.
        error = validate_extraction_params(body)
        if error:
            self._send_error(400, error)
            return

        # Allowlist bag path against catalog.
        bag_key = body.get("bag_key", "")
        catalog = discover_datasets(
            rosbag_root=cfg.get("rosbag_dir"),
            image_root=cfg.get("image_dataset_dir"),
            video_root=cfg.get("video_dataset_dir"),
        )
        try:
            bag_path = allowlist_bag_path(bag_key, catalog)
        except ValueError as exc:
            self._send_error(400, str(exc))
            return

        run_id = RunStore.new_run_id()
        run_store = self.server_instance.run_store
        frame_store = self.server_instance.frame_dataset_store
        dataset_id = RunStore.new_run_id()  # unique ID for this dataset

        output_dir = frame_store.base_dir / dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)

        params = ExtractionParams(
            bag_key=bag_key,
            bag_path=bag_path,
            image_topic=body.get("image_topic", ""),
            start_offset=body.get("start_offset", 0.0),
            end_offset=body.get("end_offset"),
            duration=body.get("duration"),
            sample_interval=body.get("sample_interval"),
            target_sample_count=body.get("target_sample_count"),
            max_frames=body.get("max_frames", 100),
            dataset_id=dataset_id,
            output_dir=str(output_dir),
        )

        script_path = str(
            pathlib.Path(__file__).parent.parent / "scripts" / "extract_bag_frames.py"
        )
        args = build_extraction_args(script_path, params)

        initial_record: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "kind": "extraction",
            "created_at": _now_iso(),
            "bag_key": bag_key,
            "bag_path": bag_path,
            "dataset_id": dataset_id,
            "image_topic": params.image_topic,
            "max_frames": params.max_frames,
            "output_dir": str(output_dir),
            "status": "starting",
        }
        run_store.save_run(run_id, initial_record)

        def _on_complete(
            rid: str,
            exit_code: Optional[int],
            was_stopped: bool,
            log_lines: list,
        ) -> None:
            status_str = "stopped" if was_stopped else (
                "completed" if exit_code == 0 else
                "ros_unavailable" if exit_code == 2 else
                "failed"
            )
            run_store.finalize_run(rid, {
                "status": status_str,
                "exit_code": exit_code,
                "completed_at": _now_iso(),
                "success": exit_code == 0 and not was_stopped,
                "log_lines": log_lines[:200],
            })

        srv = self.server_instance
        try:
            pid = srv.process_manager.start_ros_experiment(
                run_id=run_id,
                args=args,
                env=os.environ.copy(),
                on_complete=_on_complete,
            )
        except Exception as exc:
            run_store.finalize_run(run_id, {
                "status": "failed",
                "exit_code": None,
                "completed_at": _now_iso(),
                "success": False,
                "error": str(exc),
            })
            self._send_error(500, f"Failed to start extraction: {exc}")
            return

        run_store.update_run_if_status(run_id, "starting", {"status": "running", "pid": pid})
        self._send_json(202, run_store.get_run(run_id) or initial_record)

    def _api_extract_cancel(self, run_id: str) -> None:
        if not _is_safe_run_id(run_id):
            self._send_error(400, "Invalid run_id")
            return
        run_store = self.server_instance.run_store
        manifest = run_store.get_run(run_id)
        if manifest is None:
            self._send_error(404, "Run not found")
            return
        if manifest.get("kind") != "extraction":
            self._send_error(400, "Not an extraction run")
            return
        stopped = self.server_instance.process_manager.stop_ros_experiment(run_id)
        self._send_json(200, {"run_id": run_id, "stopped": stopped})

    # ── review annotation API ─────────────────────────────────────────────────

    def _api_get_reviews(self, run_id: str) -> None:
        if not _is_safe_run_id(run_id):
            self._send_error(400, "Invalid run_id")
            return
        review_store = self.server_instance.review_store
        reviews = review_store.get_reviews(run_id)
        self._send_json(200, {
            "run_id": run_id,
            "reviews": [
                {
                    "frame_index": r.frame_index,
                    "label": r.label,
                    "note": r.note,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at,
                }
                for r in reviews
            ]
        })

    def _api_upsert_review(self, run_id: str) -> None:
        if not _is_safe_run_id(run_id):
            self._send_error(400, "Invalid run_id")
            return
        body = self._read_json_body()
        if body is None:
            return
        error = validate_review(body)
        if error:
            self._send_error(400, error)
            return
        annotation = ReviewAnnotation(
            run_id=run_id,
            frame_index=int(body["frame_index"]),
            label=body["label"],
            note=body.get("note", ""),
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        review_store = self.server_instance.review_store
        try:
            review_store.upsert_review(annotation)
        except Exception as exc:
            self._send_error(500, f"Failed to save review: {exc}")
            return
        self._send_json(200, {
            "run_id": run_id,
            "frame_index": annotation.frame_index,
            "label": annotation.label,
            "note": annotation.note,
            "updated_at": annotation.updated_at,
        })

    # ── comparison API ────────────────────────────────────────────────────────

    def _api_compare_runs(self, parsed: Any) -> None:
        """Align two or more runs by source frame/timestamp.

        Query string: ?run_ids=<id1>,<id2>[,<id3>...]
        """
        from urllib.parse import parse_qs
        qs = parse_qs(parsed.query)
        run_ids_param = qs.get("run_ids", [""])[0]
        run_ids = [r.strip() for r in run_ids_param.split(",") if r.strip()]
        if len(run_ids) < 2:
            self._send_error(400, "run_ids must contain at least two run IDs")
            return
        if len(run_ids) > 8:
            self._send_error(400, "run_ids must contain at most eight run IDs")
            return
        for rid in run_ids:
            if not _is_safe_run_id(rid):
                self._send_error(400, f"Invalid run_id: {rid!r}")
                return

        run_store = self.server_instance.run_store
        manifests = {}
        for rid in run_ids:
            m = run_store.get_run(rid)
            if m is None:
                self._send_error(404, f"Run not found: {rid}")
                return
            manifests[rid] = m

        # Extract per-frame results from each run; key by (source_path or frame_index).
        per_run: Dict[str, Dict[Any, Any]] = {}
        for rid, manifest in manifests.items():
            frames: Dict[Any, Any] = {}
            for r in manifest.get("result_frames", []):
                key = r.get("source_path") or r.get("frame_index")
                frames[key] = r
            per_run[rid] = frames

        # Collect all known frame keys across all runs.
        all_keys: list = []
        seen: set = set()
        for rid in run_ids:
            for k in per_run[rid]:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        # Sort numerically where possible.
        try:
            all_keys.sort(key=lambda k: int(k) if isinstance(k, int) else 0)
        except Exception:
            all_keys.sort(key=str)

        aligned = []
        for key in all_keys:
            row: Dict[str, Any] = {"frame_key": key}
            for rid in run_ids:
                row[rid] = per_run[rid].get(key)
            aligned.append(row)

        # Build per-run summary stats.
        summaries = {}
        for rid, manifest in manifests.items():
            summaries[rid] = {
                "run_id": rid,
                "kind": manifest.get("kind"),
                "status": manifest.get("status"),
                "model": manifest.get("model"),
                "profile": manifest.get("profile"),
                "strategy": manifest.get("strategy"),
                "created_at": manifest.get("created_at"),
                "mean_latency_ms": manifest.get("mean_latency_ms"),
                "successful_frames": manifest.get("successful_frames"),
                "failed_frames": manifest.get("failed_frames"),
                "repetition_flags": manifest.get("repetition_flags"),
            }

        self._send_json(200, {
            "run_ids": run_ids,
            "aligned_frames": aligned,
            "summaries": summaries,
        })

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

_UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_RUN_RE = re.compile(
    r"^/api/runs/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:/logs)?$"
)
_RUN_REVIEWS_RE = re.compile(
    r"^/api/runs/(" + _UUID_PATTERN + r")/reviews$"
)
_FRAME_DATASET_RE = re.compile(
    r"^/api/frame-datasets/(" + _UUID_PATTERN + r")$"
)
_FRAME_IMAGE_RE = re.compile(
    r"^/api/frame-datasets/(" + _UUID_PATTERN + r")/frames/(\d{1,5})$"
)
_EXTRACT_CANCEL_RE = re.compile(
    r"^/api/extract/(" + _UUID_PATTERN + r")/cancel$"
)


# ── ROS env builder ───────────────────────────────────────────────────────────

def _build_ros_env(
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    artifact_dir: Optional[pathlib.Path] = None,
    start_worker: bool = True,
) -> Dict[str, str]:
    """Build a subprocess environment for the ROS experiment script.

    Only allowlisted parameters are passed; no arbitrary values are forwarded.
    When *artifact_dir* is supplied it is set as ARTIFACT_DIR so the script
    knows where to write result/metrics files.

    When *start_worker* is ``False`` the environment variable ``START_WORKER``
    is set to ``"false"`` and ``WORKER_SOCKET_PATH`` is forwarded from the
    console's configured socket path.  The script then connects the ROS adapter
    to the already-running standalone service instead of launching its own
    worker.
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
        "rosbag_path": "ROSBAG_PATH",
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
    # External-service mode: signal the script not to start its own worker.
    if not start_worker:
        env["START_WORKER"] = "false"
        socket_path = cfg.get("socket_path", "")
        if socket_path:
            env["WORKER_SOCKET_PATH"] = socket_path
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
        frame_dataset_store: Optional[FrameDatasetStore] = None,
        review_store: Optional[ReviewStore] = None,
    ) -> None:
        self.config = config or {}
        self.process_manager = process_manager or ProcessManager()
        runs_dir = pathlib.Path(
            self.config.get("runs_dir", pathlib.Path.home() / ".web_console" / "runs")
        )
        self.run_store = run_store or RunStore(runs_dir)
        frame_datasets_dir = pathlib.Path(
            self.config.get(
                "frame_datasets_dir",
                runs_dir / "frame_datasets",
            )
        )
        self.frame_dataset_store = frame_dataset_store or FrameDatasetStore(frame_datasets_dir)
        self.review_store = review_store or ReviewStore(runs_dir)
        self._quiet = self.config.get("quiet", False)
        # Bounded experiment coordinator: at most one active experiment at a time.
        self._active_experiment_lock = threading.Lock()
        self._active_experiment_id: Optional[str] = None

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
    active_run = srv.process_manager.active_ros_run_id() or ""
    page = _INDEX_TEMPLATE.replace("{{ACTIVE_ROS_RUN}}", html.escape(active_run))
    return page.encode("utf-8")


# ── HTML template ─────────────────────────────────────────────────────────────

_INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>edge_vlm_ros — VLM Experiment Workbench</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header>
  <h1>edge_vlm_ros — VLM Experiment Workbench</h1>
</header>
<nav aria-label="Primary navigation">
  <button data-view="dashboard" aria-label="Dashboard">Dashboard</button>
  <button data-view="models" aria-label="Models">Models</button>
  <button data-view="datasets" aria-label="Datasets">Datasets</button>
  <button data-view="frame-explorer" aria-label="Frame Explorer">Frame Explorer</button>
  <button data-view="profiles" aria-label="Task Profiles">Task Profiles</button>
  <button data-view="experiment" aria-label="Experiment">Experiment</button>
  <button data-view="runs" aria-label="Runs">Runs</button>
  <button data-view="compare" aria-label="Compare">Compare</button>
  <button data-view="diagnostics" aria-label="Diagnostics">Diagnostics</button>
</nav>
<main>

<!-- ── Dashboard view ──────────────────────────────────────────────────────── -->
<div id="view-dashboard" class="view">
  <div class="panel">
    <div class="panel-title">Service &amp; GPU Health</div>
    <div class="card-grid">
      <div class="status-card">
        <div class="card-title">Inference Service</div>
        <div id="dash-service"><span class="muted">Loading…</span></div>
      </div>
      <div class="status-card">
        <div class="card-title">GPU</div>
        <div id="dash-gpu"><span class="muted">Loading…</span></div>
      </div>
      <div class="status-card">
        <div class="card-title">Active ROS Run</div>
        <div id="dash-active"><span class="muted">None</span></div>
      </div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">Recent Runs</div>
    <div id="dash-recent"><span class="muted">Loading…</span></div>
  </div>
</div>

<!-- ── Models view ─────────────────────────────────────────────────────────── -->
<div id="view-models" class="view">
  <div class="panel">
    <div class="panel-title">Configured Model Profiles</div>
    <div id="models-list"><span class="muted">Loading…</span></div>
  </div>
</div>

<!-- ── Datasets view ───────────────────────────────────────────────────────── -->
<div id="view-datasets" class="view">
  <div id="datasets-content"><span class="muted">Loading…</span></div>
</div>

<!-- ── Experiment view ─────────────────────────────────────────────────────── -->
<div id="view-experiment" class="view">

  <!-- Standalone inference -->
  <div class="panel">
    <div class="panel-title">Standalone Inference (single image)</div>
    <form id="infer-form" enctype="multipart/form-data">
      <fieldset>
        <legend>Image &amp; Prompt</legend>
        <label>
          <span class="label-text">Image (max 64 MiB)</span>
          <input type="file" name="image" accept="image/*" required>
        </label>
        <label>
          <span class="label-text">Prompt</span>
          <input type="text" name="prompt" value="Describe the scene." size="60" required>
        </label>
      </fieldset>
      <fieldset>
        <legend>Generation Settings</legend>
        <label>
          <span class="label-text">Max tokens</span>
          <input type="number" name="max_generate_length" value="64" min="1" max="4096">
        </label>
        <label>
          <span class="label-text">Temperature</span>
          <input type="number" name="temperature" value="0.2" step="0.01" min="0" max="2">
        </label>
        <label>
          <span class="label-text">top-p</span>
          <input type="number" name="top_p" value="0.9" step="0.01" min="0.01" max="1">
        </label>
        <label>
          <span class="label-text">top-k</span>
          <input type="number" name="top_k" value="20" min="1" max="200">
        </label>
      </fieldset>
      <button type="button" id="infer-submit-btn" onclick="submitInfer()">Run Inference</button>
    </form>
    <div id="infer-result-card" style="display:none" class="result-card"></div>
    <details id="infer-raw-details" class="raw-details" style="display:none">
      <summary>Full record (JSON)</summary>
      <pre id="infer-out"></pre>
    </details>
  </div>

  <!-- Frame-sequence experiment -->
  <div class="panel">
    <div class="panel-title">Frame-Sequence Experiment (ROS-independent)</div>
    <fieldset>
      <legend>Source</legend>
      <label>
        <span class="label-text">Image paths (one per line, absolute)</span>
        <textarea id="exp-image-paths" rows="4" cols="70" placeholder="/data/frame_001.jpg&#10;/data/frame_002.jpg"></textarea>
      </label>
    </fieldset>
    <fieldset>
      <legend>Strategy &amp; Context</legend>
      <label>
        <span class="label-text">Strategy</span>
        <select id="exp-strategy">
          <option value="single_frame">single_frame — no accumulated context</option>
          <option value="single_frame_observation_history">single_frame_observation_history — rolling history</option>
        </select>
      </label>
      <label>
        <span class="label-text">History entries (0 = none)</span>
        <input type="number" id="exp-history-entries" value="0" min="0" max="256">
      </label>
      <label>
        <span class="label-text">History char budget</span>
        <input type="number" id="exp-history-chars" value="4000" min="0" max="1000000">
      </label>
    </fieldset>
    <fieldset>
      <legend>Prompt</legend>
      <label>
        <span class="label-text">System instruction</span>
        <input type="text" id="exp-system" value="You are a vision observer. Base claims on the current image." size="70">
      </label>
      <label>
        <span class="label-text">Task prompt</span>
        <input type="text" id="exp-prompt" value="Describe the scene." size="70" required>
      </label>
    </fieldset>
    <fieldset>
      <legend>Generation Settings</legend>
      <label>
        <span class="label-text">Max tokens</span>
        <input type="number" id="exp-max-gen" value="96" min="1" max="4096">
      </label>
      <label>
        <span class="label-text">Temperature</span>
        <input type="number" id="exp-temperature" value="0.2" step="0.01" min="0" max="2">
      </label>
      <label>
        <span class="label-text">top-p</span>
        <input type="number" id="exp-top-p" value="0.9" step="0.01" min="0.01" max="1">
      </label>
      <label>
        <span class="label-text">top-k</span>
        <input type="number" id="exp-top-k" value="20" min="1" max="200">
      </label>
      <label>
        <span class="label-text">Timeout (s)</span>
        <input type="number" id="exp-timeout" value="120" min="1" max="3600">
      </label>
    </fieldset>
    <fieldset>
      <legend>Reproducibility</legend>
      <label>
        <span class="label-text">Notes</span>
        <input type="text" id="exp-notes" size="70" placeholder="Optional free-form notes">
      </label>
    </fieldset>
    <button type="button" id="exp-submit-btn" onclick="submitExperiment()">Submit Experiment</button>
    <div id="exp-result"></div>
  </div>

  <!-- ROS experiment -->
  <div class="panel">
    <div class="panel-title">ROS Image-Proc Experiment (rosbag playback)</div>
    <div id="ros-active" class="active-run-banner" style="display:none">
      Active run: <span id="ros-active-id"></span>
      <button class="danger small" onclick="stopRosRun(document.getElementById('ros-active-id').textContent)">Stop</button>
    </div>
    <div id="ros-selected-bag" class="selected-bag-status" style="display:none"></div>
    <fieldset>
      <legend>ROS Source &amp; Prompt</legend>
      <label>
        <span class="label-text">Image topic</span>
        <input id="ros-topic" type="text" value="/hawk_0_left_rgb_image" size="40">
      </label>
      <label>
        <span class="label-text">Prompt override (optional)</span>
        <input id="ros-prompt" type="text" value="" placeholder="(optional override)" size="60">
      </label>
    </fieldset>
    <fieldset>
      <legend>Generation &amp; History</legend>
      <label>
        <span class="label-text">Max tokens</span>
        <input id="ros-max-gen" type="number" value="64" min="1" max="4096">
      </label>
      <label>
        <span class="label-text">Delivery mode</span>
        <select id="ros-delivery">
          <option value="inline">inline</option>
          <option value="structured">structured</option>
        </select>
      </label>
      <label>
        <span class="label-text">Obs. history entries</span>
        <input id="ros-hist-entries" type="number" value="0" min="0" max="256">
      </label>
      <label>
        <span class="label-text">Obs. history chars</span>
        <input id="ros-hist-chars" type="number" value="0" min="0" max="1000000">
      </label>
    </fieldset>
    <fieldset>
      <legend>Timing</legend>
      <label>
        <span class="label-text">Playback duration (s)</span>
        <input id="ros-playback" type="number" value="20" min="1" max="3600">
      </label>
      <label>
        <span class="label-text">Result timeout (s)</span>
        <input id="ros-timeout" type="number" value="120" min="1" max="3600">
      </label>
      <label>
        <span class="label-text">Required results</span>
        <input id="ros-required" type="number" value="1" min="1" max="100">
      </label>
    </fieldset>
    <button type="button" onclick="startRos()">Start ROS Experiment</button>
    <div id="ros-start-out" style="display:none" class="muted"></div>
    <div id="ros-logs-area" style="display:none">
      <div class="panel-title" style="margin-top:0.75rem">Live Logs</div>
      <pre id="ros-logs" style="max-height:300px"></pre>
    </div>
    <details id="ros-raw-details" class="raw-details" style="display:none">
      <summary>Raw start response (JSON)</summary>
      <pre id="ros-out"></pre>
    </details>
  </div>
</div>

<!-- ── Runs view ────────────────────────────────────────────────────────────── -->
<div id="view-runs" class="view">
  <div class="panel">
    <div class="panel-title">Run Catalog
      <button class="secondary small" onclick="_loadRuns()" style="margin-left:auto">Refresh</button>
    </div>
    <div id="runs-list"><span class="muted">Loading…</span></div>
  </div>
  <div id="run-detail" style="display:none" class="panel">
    <div class="panel-title">Run Detail</div>
    <div id="run-detail-reviews" class="review-panel">
      <div class="panel-title">Review Annotations</div>
      <div id="review-list"><span class="muted">Select a run to view reviews.</span></div>
    </div>
  </div>
</div>

<!-- ── Diagnostics view ────────────────────────────────────────────────────── -->
<div id="view-diagnostics" class="view">
  <div class="panel">
    <div class="panel-title">Live Status
      <button class="secondary small" onclick="loadDiagStatus()" style="margin-left:auto">Refresh</button>
    </div>
    <pre id="diag-status-pre">Loading…</pre>
  </div>
  <div class="panel">
    <div class="panel-title">Run Manifests
      <button class="secondary small" onclick="loadDiagRuns()" style="margin-left:auto">Refresh</button>
    </div>
    <pre id="diag-runs-pre">Click Refresh to load.</pre>
  </div>
  <div class="panel">
    <div class="panel-title">Models (raw JSON)
      <button class="secondary small" onclick="loadDiagModels()" style="margin-left:auto">Refresh</button>
    </div>
    <pre id="diag-models-pre">Click Refresh to load.</pre>
  </div>
  <div class="panel">
    <div class="panel-title">Datasets (raw JSON)
      <button class="secondary small" onclick="loadDiagDatasets()" style="margin-left:auto">Refresh</button>
    </div>
    <pre id="diag-datasets-pre">Click Refresh to load.</pre>
  </div>
</div>

<!-- ── Frame Explorer view ──────────────────────────────────────────────────── -->
<div id="view-frame-explorer" class="view">
  <div class="panel">
    <div class="panel-title">Frame Datasets
      <button class="secondary small" onclick="_loadFrameExplorer()" style="margin-left:auto">Refresh</button>
    </div>
    <div id="frame-dataset-list"><span class="muted">Loading…</span></div>
  </div>
  <div class="panel" id="frame-explorer-viewer" style="display:none">
    <div class="panel-title">Frame Viewer
      <button class="secondary small" onclick="_framePrev()">&#8592; Prev</button>
      <button class="secondary small" onclick="_frameNext()">Next &#8594;</button>
    </div>
    <div id="frame-preview-area"></div>
    <div id="frame-metadata" class="frame-meta"></div>
    <div id="frame-thumbnail-strip" class="thumb-strip"></div>
    <div id="frame-review-ui" class="review-panel"></div>
  </div>
</div>

<!-- ── Task Profiles view ──────────────────────────────────────────────────── -->
<div id="view-profiles" class="view">
  <div class="panel">
    <div class="panel-title">Task Profiles (warehouse awareness &amp; custom)
      <button class="secondary small" onclick="_loadProfiles()" style="margin-left:auto">Refresh</button>
    </div>
    <div id="profiles-list"><span class="muted">Loading…</span></div>
  </div>
</div>

<!-- ── Compare view ────────────────────────────────────────────────────────── -->
<div id="view-compare" class="view">
  <div class="panel">
    <div class="panel-title">Compare Runs</div>
    <div id="compare-form" class="compare-form-row">
      <label>Run ID 1: <input id="compare-run-id-1" type="text" placeholder="UUID" size="40"></label>
      <label>Run ID 2: <input id="compare-run-id-2" type="text" placeholder="UUID" size="40"></label>
      <button onclick="_runCompare()">Compare</button>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">Results</div>
    <div id="compare-results"><span class="muted">Enter run IDs above to compare.</span></div>
  </div>
</div>

</main>
<script src="/static/app.js"></script>
<script>
  (function() {
    var activeRos = "{{ACTIVE_ROS_RUN}}";
    if (activeRos) { showActiveRun(activeRos); }
  })();
</script>
</body>
</html>
"""



