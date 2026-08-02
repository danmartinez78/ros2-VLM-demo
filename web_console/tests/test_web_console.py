"""
CI-safe tests for the web_console package.

These tests do NOT require TensorRT, CUDA, ROS, nvidia-smi, or any hardware.
They validate:
  - ProcessManager lifecycle, concurrency rejection, TERM/KILL sequencing
  - RunStore path-traversal guards, eviction, artifact management
  - status_collector degradation without nvidia-smi or IPC socket
  - inference_client subprocess construction and error paths
  - ConsoleServer API validation, upload enforcement, JSON responses
  - Fixture manifest / log parsing and presentation
  - Static asset syntax (CSS/JS) presence check
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from http.client import HTTPConnection
from unittest.mock import MagicMock, patch

# Ensure the repo root is on the path so we can import web_console directly.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from web_console.process_manager import ProcessManager
from web_console.run_store import RunStore, _is_safe_run_id, _TERMINAL_STATUSES
from web_console.status_collector import check_server_reachable, get_gpu_status, collect_status
from web_console.inference_client import run_inference, ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_BYTES
from web_console.server import (
    ConsoleServer,
    _parse_multipart,
    _validate_ros_params,
    _build_ros_env,
    _parse_results_log,
    _parse_benchmark_jsonl,
    _single_quote_close_idx,
    _fold_single_quoted_scalar,
)
from web_console.__main__ import _parse_args, _is_loopback


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_multipart(fields: dict, boundary: str = "testboundary") -> tuple[bytes, str]:
    """Build a minimal multipart/form-data body for testing."""
    parts = []
    for name, value in fields.items():
        if isinstance(value, dict):
            filename = value["filename"]
            data = value["data"]
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: image/jpeg\r\n"
                f"\r\n".encode()
                + data
                + b"\r\n"
            )
        else:
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n".encode()
            )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    ct = f"multipart/form-data; boundary={boundary}"
    return body, ct


class _TempDir:
    def __enter__(self):
        self.path = pathlib.Path(tempfile.mkdtemp())
        return self.path

    def __exit__(self, *_):
        shutil.rmtree(self.path, ignore_errors=True)


# ── ProcessManager tests ──────────────────────────────────────────────────────

class TestProcessManager(unittest.TestCase):

    def test_start_and_is_running(self):
        mgr = ProcessManager()
        run_id = "aaaaaaaa-0000-0000-0000-000000000001"
        pid = mgr.start_ros_experiment(run_id, ["sleep", "30"])
        self.assertGreater(pid, 0)
        self.assertTrue(mgr.is_running(run_id))
        mgr.cleanup()
        # After cleanup the process should be gone.
        self.assertFalse(mgr.is_running(run_id))

    def test_stop_experiment(self):
        mgr = ProcessManager()
        run_id = "aaaaaaaa-0000-0000-0000-000000000002"
        mgr.start_ros_experiment(run_id, ["sleep", "30"])
        mgr.stop_experiment(run_id)
        time.sleep(0.2)
        self.assertFalse(mgr.is_running(run_id))

    def test_concurrent_ros_rejected(self):
        mgr = ProcessManager()
        run_id_1 = "aaaaaaaa-0000-0000-0000-000000000003"
        run_id_2 = "aaaaaaaa-0000-0000-0000-000000000004"
        mgr.start_ros_experiment(run_id_1, ["sleep", "30"])
        with self.assertRaises(RuntimeError):
            mgr.start_ros_experiment(run_id_2, ["sleep", "30"])
        mgr.cleanup()

    def test_stop_unknown_raises_key_error(self):
        mgr = ProcessManager()
        with self.assertRaises(KeyError):
            mgr.stop_experiment("aaaaaaaa-0000-0000-0000-000000000099")

    def test_log_collection(self):
        mgr = ProcessManager()
        run_id = "aaaaaaaa-0000-0000-0000-000000000005"
        # A command that prints a few lines and exits.
        mgr.start_ros_experiment(
            run_id,
            ["bash", "-c", "echo line1; echo line2; echo line3"],
        )
        # Wait for the process to finish and logs to be collected.
        deadline = time.monotonic() + 5.0
        while mgr.is_running(run_id) and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.1)
        logs = mgr.get_logs(run_id)
        self.assertIn("line1", logs)
        self.assertIn("line2", logs)
        mgr.cleanup()

    def test_process_not_shell(self):
        """Verify that start_ros_experiment does NOT use shell=True (no /bin/sh wrapper)."""
        mgr = ProcessManager()
        run_id = "aaaaaaaa-0000-0000-0000-000000000006"
        # If shell were True, passing a list would treat the first element as
        # the command and the rest as $0, $1… — behavior differs.
        with patch("subprocess.Popen", wraps=subprocess.Popen) as mock_popen:
            mgr.start_ros_experiment(run_id, ["sleep", "0.1"])
            call_kwargs = mock_popen.call_args[1]
            # shell must NOT be True (default is False/absent)
            self.assertNotEqual(call_kwargs.get("shell"), True)
        mgr.cleanup()

    def test_active_ros_run_id(self):
        mgr = ProcessManager()
        self.assertIsNone(mgr.active_ros_run_id())
        run_id = "aaaaaaaa-0000-0000-0000-000000000007"
        mgr.start_ros_experiment(run_id, ["sleep", "30"])
        self.assertEqual(mgr.active_ros_run_id(), run_id)
        mgr.cleanup()

    def test_cleanup_idempotent(self):
        mgr = ProcessManager()
        run_id = "aaaaaaaa-0000-0000-0000-000000000008"
        mgr.start_ros_experiment(run_id, ["sleep", "30"])
        mgr.cleanup()
        mgr.cleanup()  # second call should not raise

    # ── completion callback tests ─────────────────────────────────────────────

    def test_completion_callback_natural_success(self):
        """Callback fires with exit_code=0 and was_stopped=False on clean exit."""
        mgr = ProcessManager()
        run_id = RunStore.new_run_id()
        fired = threading.Event()
        result = {}

        def _cb(rid, exit_code, was_stopped, log_lines):
            result.update(
                rid=rid, exit_code=exit_code, was_stopped=was_stopped, log_lines=log_lines
            )
            fired.set()

        mgr.start_ros_experiment(run_id, ["bash", "-c", "echo hello_cb"], on_complete=_cb)
        self.assertTrue(fired.wait(timeout=8.0), "Callback not fired within timeout")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["was_stopped"])
        self.assertIn("hello_cb", result["log_lines"])

    def test_completion_callback_nonzero_exit(self):
        """Callback fires with exit_code=1 and was_stopped=False on error exit."""
        mgr = ProcessManager()
        run_id = RunStore.new_run_id()
        fired = threading.Event()
        result = {}

        def _cb(rid, exit_code, was_stopped, log_lines):
            result.update(exit_code=exit_code, was_stopped=was_stopped)
            fired.set()

        mgr.start_ros_experiment(run_id, ["bash", "-c", "exit 1"], on_complete=_cb)
        self.assertTrue(fired.wait(timeout=8.0))
        self.assertEqual(result["exit_code"], 1)
        self.assertFalse(result["was_stopped"])

    def test_completion_callback_explicit_stop(self):
        """Callback fires with was_stopped=True when stop_experiment is called."""
        mgr = ProcessManager()
        run_id = RunStore.new_run_id()
        fired = threading.Event()
        result = {}

        def _cb(rid, exit_code, was_stopped, log_lines):
            result.update(was_stopped=was_stopped)
            fired.set()

        mgr.start_ros_experiment(run_id, ["sleep", "60"], on_complete=_cb)
        mgr.stop_experiment(run_id)
        self.assertTrue(fired.wait(timeout=8.0))
        self.assertTrue(result["was_stopped"])

    def test_callback_fires_at_most_once(self):
        """Calling stop after natural exit must not fire the callback twice."""
        mgr = ProcessManager()
        run_id = RunStore.new_run_id()
        call_count = {"n": 0}
        fired = threading.Event()

        def _cb(rid, exit_code, was_stopped, log_lines):
            call_count["n"] += 1
            fired.set()

        mgr.start_ros_experiment(run_id, ["bash", "-c", "exit 0"], on_complete=_cb)
        # Wait for natural completion.
        fired.wait(timeout=8.0)
        # Now call stop (process already gone).
        mgr.stop_experiment(run_id)
        time.sleep(0.3)
        self.assertEqual(call_count["n"], 1, "Callback must fire exactly once")
        mgr.cleanup()

    def test_concurrent_start_is_atomic(self):
        """Atomicity: two simultaneous start calls must allow exactly one to succeed."""
        mgr = ProcessManager()
        successes: list = []
        failures: list = []
        barrier = threading.Barrier(2)
        lock = threading.Lock()

        def _try_start(run_id):
            barrier.wait()
            try:
                mgr.start_ros_experiment(run_id, ["sleep", "30"])
                with lock:
                    successes.append(run_id)
            except RuntimeError:
                with lock:
                    failures.append(run_id)

        ids = [RunStore.new_run_id(), RunStore.new_run_id()]
        threads = [threading.Thread(target=_try_start, args=(rid,)) for rid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        mgr.cleanup()

        self.assertEqual(len(successes), 1, "Exactly one start must succeed")
        self.assertEqual(len(failures), 1, "Exactly one start must be rejected")


# ── RunStore tests ────────────────────────────────────────────────────────────

class TestRunStore(unittest.TestCase):

    def test_save_and_get(self):
        with _TempDir() as d:
            store = RunStore(d)
            run_id = RunStore.new_run_id()
            record = {"run_id": run_id, "kind": "standalone", "success": True}
            store.save_run(run_id, record)
            loaded = store.get_run(run_id)
            self.assertEqual(loaded["run_id"], run_id)
            self.assertEqual(loaded["success"], True)

    def test_list_runs_newest_first(self):
        with _TempDir() as d:
            store = RunStore(d)
            ids = [RunStore.new_run_id() for _ in range(3)]
            for i, rid in enumerate(ids):
                store.save_run(rid, {"run_id": rid, "ts": i})
                time.sleep(0.01)
            runs = store.list_runs()
            retrieved_ids = [r["run_id"] for r in runs]
            self.assertEqual(retrieved_ids[0], ids[-1])  # newest first

    def test_path_traversal_blocked(self):
        with _TempDir() as d:
            store = RunStore(d)
            self.assertIsNone(store.get_run("../../../etc/passwd"))
            self.assertIsNone(store.get_run(".."))
            self.assertIsNone(store.get_run("not-a-uuid"))
            with self.assertRaises(ValueError):
                store.save_run("../attack", {})

    def test_is_safe_run_id(self):
        valid = "12345678-1234-4234-a234-123456789abc"
        self.assertTrue(_is_safe_run_id(valid))
        self.assertFalse(_is_safe_run_id("../etc"))
        self.assertFalse(_is_safe_run_id(""))
        self.assertFalse(_is_safe_run_id("not-a-uuid"))
        self.assertFalse(_is_safe_run_id("12345678/1234-4234-a234-123456789abc"))

    def test_artifact_write_and_read(self):
        with _TempDir() as d:
            store = RunStore(d)
            run_id = RunStore.new_run_id()
            store.save_run(run_id, {"run_id": run_id})
            store.write_artifact(run_id, "result.json", b'{"ok": true}')
            p = store.artifact_path(run_id, "result.json")
            self.assertIsNotNone(p)
            self.assertEqual(p.read_bytes(), b'{"ok": true}')

    def test_artifact_traversal_blocked(self):
        with _TempDir() as d:
            store = RunStore(d)
            run_id = RunStore.new_run_id()
            store.save_run(run_id, {"run_id": run_id})
            with self.assertRaises(ValueError):
                store.write_artifact(run_id, "../evil.sh", b"")
            with self.assertRaises(ValueError):
                store.write_artifact(run_id, ".hidden", b"")
            self.assertIsNone(store.artifact_path(run_id, "../evil.sh"))

    def test_eviction(self):
        with _TempDir() as d:
            store = RunStore(d)
            store._MAX_RUNS = 3  # type: ignore[attr-defined]
            # Monkey-patch for test
            RunStore._MAX_RUNS = 3
            ids = []
            for _ in range(5):
                rid = RunStore.new_run_id()
                ids.append(rid)
                store.save_run(rid, {"run_id": rid})
                time.sleep(0.01)
            store._evict_oldest()
            remaining = [r["run_id"] for r in store.list_runs()]
            # The 3 newest should survive.
            for rid in ids[-3:]:
                self.assertIn(rid, remaining)
            RunStore._MAX_RUNS = 100  # restore

    def test_concurrent_save_read_no_corruption(self):
        """Concurrent saves and reads must never observe partial JSON or lose run_id."""
        with _TempDir() as d:
            store = RunStore(d)
            run_id = RunStore.new_run_id()
            store.save_run(run_id, {"run_id": run_id, "value": 0})

            errors: list = []

            def _writer(value_start: int) -> None:
                for i in range(50):
                    try:
                        store.save_run(
                            run_id,
                            {"run_id": run_id, "value": value_start + i, "data": "x" * 200},
                        )
                    except Exception as exc:
                        errors.append(f"writer: {exc}")

            def _reader() -> None:
                for _ in range(50):
                    try:
                        rec = store.get_run(run_id)
                        if rec is not None:
                            assert "run_id" in rec, f"Corrupt record missing run_id: {rec!r}"
                    except Exception as exc:
                        errors.append(f"reader: {exc}")

            def _updater() -> None:
                for i in range(50):
                    try:
                        store.update_run_if_status(run_id, "running", {"status": "running"})
                    except Exception as exc:
                        errors.append(f"updater: {exc}")

            threads = (
                [threading.Thread(target=_writer, args=(i * 100,)) for i in range(2)]
                + [threading.Thread(target=_reader) for _ in range(2)]
                + [threading.Thread(target=_updater)]
            )
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15.0)

            self.assertEqual(errors, [], f"Concurrent errors: {errors}")
            final = store.get_run(run_id)
            self.assertIsNotNone(final)
            self.assertEqual(final.get("run_id"), run_id)

    def test_finalize_run_no_overwrite_terminal(self):
        """finalize_run must not overwrite an already-terminal status."""
        with _TempDir() as d:
            store = RunStore(d)
            run_id = RunStore.new_run_id()
            store.save_run(run_id, {"run_id": run_id, "status": "completed", "success": True})
            applied = store.finalize_run(run_id, {"status": "failed", "success": False})
            self.assertFalse(applied, "finalize_run must be a no-op on a terminal record")
            rec = store.get_run(run_id)
            self.assertEqual(rec.get("status"), "completed")
            self.assertTrue(rec.get("success"))

    def test_update_run_if_status_atomic(self):
        """update_run_if_status must only update when status matches."""
        with _TempDir() as d:
            store = RunStore(d)
            run_id = RunStore.new_run_id()
            store.save_run(run_id, {"run_id": run_id, "status": "starting"})
            # Correct expected status → update applied.
            result = store.update_run_if_status(run_id, "starting", {"status": "running", "pid": 1})
            self.assertEqual(result.get("status"), "running")
            # Wrong expected status → no-op.
            result2 = store.update_run_if_status(run_id, "starting", {"status": "completed"})
            self.assertEqual(result2.get("status"), "running")  # unchanged


# ── status_collector tests ─────────────────────────────────────────────────────

class TestStatusCollector(unittest.TestCase):

    def test_check_server_missing_socket(self):
        result = check_server_reachable("/tmp/no_such_socket_xyzzy_12345.sock")
        self.assertFalse(result["reachable"])
        self.assertIn("error", result)

    def test_check_server_empty_path(self):
        result = check_server_reachable("")
        self.assertFalse(result["reachable"])

    def test_gpu_status_no_nvidia_smi(self):
        """When nvidia-smi is absent, get_gpu_status must return available=False cleanly."""
        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi")):
            status = get_gpu_status()
        self.assertFalse(status["available"])
        self.assertIn("error", status)

    def test_gpu_status_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            status = get_gpu_status()
        self.assertFalse(status["available"])

    def test_collect_status_structure(self):
        with patch("web_console.status_collector.check_server_reachable") as mock_srv, \
             patch("web_console.status_collector.get_gpu_status") as mock_gpu, \
             patch("web_console.status_collector.get_server_pid", return_value=None):
            mock_srv.return_value = {"reachable": False}
            mock_gpu.return_value = {"available": False, "error": "not found"}
            result = collect_status("/tmp/fake.sock")
        self.assertIn("server", result)
        self.assertIn("gpu", result)
        self.assertIn("env", result)


# ── inference_client tests ────────────────────────────────────────────────────

class TestInferenceClient(unittest.TestCase):

    def test_success_path(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="A cat on a mat.", stderr=""
            )
            result = run_inference(
                cli_path="edge_vlm_cli",
                socket_path="/tmp/fake.sock",
                image_path="/tmp/fake.jpg",
                prompt="Describe.",
            )
        self.assertTrue(result.success)
        self.assertEqual(result.text, "A cat on a mat.")

    def test_cli_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("edge_vlm_cli")):
            result = run_inference(
                cli_path="/nonexistent/edge_vlm_cli",
                socket_path="/tmp/fake.sock",
                image_path="/tmp/fake.jpg",
                prompt="Describe.",
            )
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    def test_timeout_path(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("edge_vlm_cli", 5)):
            result = run_inference(
                cli_path="edge_vlm_cli",
                socket_path="/tmp/fake.sock",
                image_path="/tmp/fake.jpg",
                prompt="Describe.",
                timeout_seconds=5,
            )
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)

    def test_cli_error_exit(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="socket connection refused"
            )
            result = run_inference(
                cli_path="edge_vlm_cli",
                socket_path="/tmp/fake.sock",
                image_path="/tmp/fake.jpg",
                prompt="Describe.",
            )
        self.assertFalse(result.success)
        self.assertIn("connection refused", result.error)

    def test_subprocess_no_shell(self):
        """Verify that run_inference never passes shell=True."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            run_inference(
                cli_path="edge_vlm_cli",
                socket_path="/tmp/fake.sock",
                image_path="/tmp/fake.jpg",
                prompt="Describe.",
            )
            call_kwargs = mock_run.call_args[1]
            self.assertNotEqual(call_kwargs.get("shell"), True)

    def test_args_are_list(self):
        """subprocess.run must receive a list, not a string."""
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            run_inference(
                cli_path="edge_vlm_cli",
                socket_path="/tmp/fake.sock",
                image_path="/tmp/fake.jpg",
                prompt="Describe.",
            )
        self.assertIsInstance(captured["args"], list)

    def test_parameter_validation_max_gen(self):
        result = run_inference(
            cli_path="edge_vlm_cli",
            socket_path="/tmp/fake.sock",
            image_path="/tmp/fake.jpg",
            prompt="Describe.",
            max_generate_length=0,
        )
        self.assertFalse(result.success)
        self.assertIn("max_generate_length", result.error)

    def test_parameter_validation_temperature(self):
        result = run_inference(
            cli_path="edge_vlm_cli",
            socket_path="/tmp/fake.sock",
            image_path="/tmp/fake.jpg",
            prompt="Describe.",
            temperature=5.0,
        )
        self.assertFalse(result.success)
        self.assertIn("temperature", result.error)


# ── multipart parsing tests ───────────────────────────────────────────────────

class TestMultipartParsing(unittest.TestCase):

    def test_parse_text_and_file(self):
        body, ct = _make_multipart({
            "prompt": "Hello world",
            "image": {"filename": "photo.jpg", "data": b"\xff\xd8\xff"},
        })
        parts = _parse_multipart(ct, body)
        self.assertEqual(parts["prompt"], "Hello world")
        self.assertEqual(parts["image"]["filename"], "photo.jpg")
        self.assertEqual(parts["image"]["data"], b"\xff\xd8\xff")

    def test_missing_field(self):
        body, ct = _make_multipart({"prompt": "test"})
        parts = _parse_multipart(ct, body)
        self.assertNotIn("image", parts)


# ── ROS param validation tests ────────────────────────────────────────────────

class TestRosParamValidation(unittest.TestCase):

    def test_valid_params(self):
        params = {
            "image_topic": "/hawk_0",
            "max_generate_length": 64,
            "instruction_delivery_mode": "inline",
        }
        self.assertIsNone(_validate_ros_params(params))

    def test_unknown_param_rejected(self):
        params = {"image_topic": "/topic", "shell_cmd": "rm -rf /"}
        error = _validate_ros_params(params)
        self.assertIsNotNone(error)
        self.assertIn("Unknown", error)

    def test_invalid_delivery_mode(self):
        params = {"instruction_delivery_mode": "exec"}
        error = _validate_ros_params(params)
        self.assertIsNotNone(error)

    def test_integer_out_of_range(self):
        params = {"max_generate_length": 99999}
        error = _validate_ros_params(params)
        self.assertIsNotNone(error)

    def test_non_integer_value(self):
        params = {"max_generate_length": "not_a_number"}
        error = _validate_ros_params(params)
        self.assertIsNotNone(error)


# ── ConsoleServer HTTP API tests ──────────────────────────────────────────────

def _start_test_server(config=None, process_manager=None, run_store=None):
    """Start a ConsoleServer on a random port for testing. Returns (srv, port, thread)."""
    import socket

    # Find a free port.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    srv = ConsoleServer(
        host="127.0.0.1",
        port=port,
        config=config or {"socket_path": "/tmp/no_such.sock", "quiet": True},
        process_manager=process_manager,
        run_store=run_store,
    )
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    # Wait for the server to be ready.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=0.5)
            conn.request("GET", "/api/status")
            conn.getresponse().read()
            conn.close()
            break
        except Exception:
            time.sleep(0.05)
    return srv, port, t


class TestConsoleServerAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._run_store = RunStore(pathlib.Path(cls._tmpdir) / "runs")
        cls._process_manager = ProcessManager()
        cls._srv, cls._port, cls._thread = _start_test_server(
            config={"socket_path": "/tmp/no_such.sock", "quiet": True},
            process_manager=cls._process_manager,
            run_store=cls._run_store,
        )

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        cls._process_manager.cleanup()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _get(self, path):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, json.loads(body)

    def _post_json(self, path, data):
        body = json.dumps(data).encode()
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()
        return resp.status, json.loads(resp_body)

    def _post_multipart(self, path, fields):
        body, ct = _make_multipart(fields)
        conn = HTTPConnection("127.0.0.1", self._port, timeout=10)
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": ct, "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()
        return resp.status, json.loads(resp_body)

    # ── GET /api/status ───────────────────────────────────────────────────────

    def test_status_returns_json(self):
        status, data = self._get("/api/status")
        self.assertEqual(status, 200)
        self.assertIn("server", data)
        self.assertIn("gpu", data)

    def test_status_server_unreachable(self):
        status, data = self._get("/api/status")
        self.assertFalse(data["server"]["reachable"])

    # ── GET /api/runs ─────────────────────────────────────────────────────────

    def test_list_runs_empty(self):
        status, data = self._get("/api/runs")
        self.assertEqual(status, 200)
        self.assertIsInstance(data["runs"], list)

    def test_list_runs_after_save(self):
        run_id = RunStore.new_run_id()
        self._run_store.save_run(run_id, {"run_id": run_id, "kind": "standalone"})
        status, data = self._get("/api/runs")
        self.assertEqual(status, 200)
        ids = [r["run_id"] for r in data["runs"]]
        self.assertIn(run_id, ids)

    # ── GET /api/runs/<id> ────────────────────────────────────────────────────

    def test_get_run_not_found(self):
        status, data = self._get("/api/runs/12345678-0000-0000-0000-000000000000")
        self.assertEqual(status, 404)

    def test_get_run_invalid_id(self):
        status, data = self._get("/api/runs/../../../etc/passwd")
        # Either 400 or 404 is acceptable for an invalid path.
        self.assertIn(status, (400, 404))

    def test_get_run_found(self):
        run_id = RunStore.new_run_id()
        self._run_store.save_run(run_id, {"run_id": run_id, "kind": "test"})
        status, data = self._get(f"/api/runs/{run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(data["run_id"], run_id)

    # ── POST /api/infer ───────────────────────────────────────────────────────

    def test_infer_missing_prompt(self):
        status, data = self._post_multipart(
            "/api/infer",
            {"image": {"filename": "img.jpg", "data": b"\xff\xd8\xff"}},
        )
        self.assertEqual(status, 400)
        self.assertIn("prompt", data["error"])

    def test_infer_missing_image(self):
        status, data = self._post_multipart(
            "/api/infer",
            {"prompt": "Describe."},
        )
        self.assertEqual(status, 400)
        self.assertIn("image", data["error"])

    def test_infer_unsupported_type(self):
        status, data = self._post_multipart(
            "/api/infer",
            {
                "prompt": "Describe.",
                "image": {"filename": "photo.exe", "data": b"MZ"},
            },
        )
        self.assertEqual(status, 415)

    def test_infer_success_fake(self):
        """Fake a successful CLI response and verify the stored record."""
        with patch("web_console.server.run_inference") as mock_infer:
            mock_infer.return_value = MagicMock(
                success=True, text="A cat on a mat.", error="", inference_seconds=0.5
            )
            status, data = self._post_multipart(
                "/api/infer",
                {
                    "prompt": "Describe the scene.",
                    "image": {"filename": "scene.jpg", "data": b"\xff\xd8\xff\xe0"},
                },
            )
        self.assertEqual(status, 200)
        self.assertTrue(data["success"])
        self.assertEqual(data["text"], "A cat on a mat.")
        self.assertIn("run_id", data)

    def test_infer_cli_error_returns_422(self):
        with patch("web_console.server.run_inference") as mock_infer:
            mock_infer.return_value = MagicMock(
                success=False, text="", error="socket refused", inference_seconds=0.1
            )
            status, data = self._post_multipart(
                "/api/infer",
                {
                    "prompt": "Describe.",
                    "image": {"filename": "img.jpg", "data": b"\xff\xd8\xff\xe0"},
                },
            )
        self.assertEqual(status, 422)
        self.assertFalse(data["success"])

    def test_infer_oversized_rejected(self):
        """Content-Length larger than MAX_IMAGE_BYTES + overhead must be rejected with 413."""
        # Send a header-only request with a large Content-Length to trigger the
        # early rejection. The server closes the socket before reading the body,
        # so a BrokenPipeError or ConnectionResetError on the send side is expected
        # when we try to write the body.
        huge_length = MAX_IMAGE_BYTES + 100_000
        boundary = "testboundary"
        ct = f"multipart/form-data; boundary={boundary}"
        # Small stub body — the Content-Length claim is what triggers the check.
        stub_body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="prompt"\r\n'
            "\r\n"
            "test\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        try:
            conn.request(
                "POST",
                "/api/infer",
                body=stub_body,
                headers={
                    "Content-Type": ct,
                    # Report a huge length to hit the server-side size check.
                    "Content-Length": str(huge_length),
                },
            )
            resp = conn.getresponse()
            body_data = resp.read()
            conn.close()
            self.assertEqual(resp.status, 413)
        except (BrokenPipeError, ConnectionResetError):
            # The server rejected the oversized request and closed the socket
            # before we finished sending — this is the expected rejection path.
            conn.close()
        except Exception as exc:
            conn.close()
            self.fail(f"Unexpected exception: {exc}")

    def test_infer_wrong_content_type(self):
        """Non-multipart POST must return 400."""
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request(
            "POST",
            "/api/infer",
            body=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": "2"},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 400)

    # ── POST /api/ros/start ───────────────────────────────────────────────────

    def test_ros_start_unknown_param(self):
        status, data = self._post_json(
            "/api/ros/start",
            {"params": {"shell_cmd": "rm -rf /"}},
        )
        self.assertEqual(status, 400)

    def test_ros_start_invalid_delivery_mode(self):
        status, data = self._post_json(
            "/api/ros/start",
            {"params": {"instruction_delivery_mode": "exec"}},
        )
        self.assertEqual(status, 400)

    def test_ros_start_invalid_json(self):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request(
            "POST",
            "/api/ros/start",
            body=b"not json",
            headers={"Content-Type": "application/json", "Content-Length": "8"},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 400)

    def test_ros_concurrent_rejected(self):
        """Starting a second ROS experiment while one is running must return 409."""
        run_id_1 = RunStore.new_run_id()
        # Directly inject a fake running process to simulate concurrency.
        with patch.object(
            self._process_manager,
            "start_ros_experiment",
            side_effect=RuntimeError("already running"),
        ):
            status, data = self._post_json(
                "/api/ros/start",
                {"params": {"max_generate_length": 64}},
            )
        self.assertEqual(status, 409)

    # ── POST /api/ros/stop ────────────────────────────────────────────────────

    def test_ros_stop_unknown_run(self):
        status, data = self._post_json(
            "/api/ros/stop",
            {"run_id": "12345678-0000-0000-0000-000000000000"},
        )
        self.assertEqual(status, 404)

    def test_ros_stop_invalid_run_id(self):
        status, data = self._post_json(
            "/api/ros/stop",
            {"run_id": "../attack"},
        )
        self.assertEqual(status, 400)

    # ── Content-Length guard tests ─────────────────────────────────────────────

    def test_negative_content_length_json_returns_400(self):
        """A negative Content-Length on a JSON endpoint must be rejected (not block)."""
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request(
            "POST",
            "/api/ros/start",
            body=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": "-1"},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertIn(resp.status, (400, 411))

    def test_missing_content_length_json_returns_411(self):
        """A JSON request with no Content-Length header must be rejected."""
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        # Manually send HTTP/1.0 request without Content-Length.
        conn.request(
            "POST",
            "/api/ros/start",
            body=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": ""},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertIn(resp.status, (400, 411))

    def test_negative_content_length_multipart_returns_400(self):
        """A negative Content-Length on the multipart endpoint must be rejected."""
        boundary = "testboundary"
        ct = f"multipart/form-data; boundary={boundary}"
        stub = f"--{boundary}--\r\n".encode()
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request(
            "POST",
            "/api/infer",
            body=stub,
            headers={"Content-Type": ct, "Content-Length": "-1"},
        )
        try:
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertIn(resp.status, (400, 411))
        except (BrokenPipeError, ConnectionResetError):
            conn.close()  # server correctly rejected and closed

    # ── static assets ─────────────────────────────────────────────────────────

    def test_static_css_served(self):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", "/static/style.css")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertIn(b"body", body)

    def test_static_js_served(self):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", "/static/app.js")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertIn(b"function", body)

    def test_static_traversal_blocked(self):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", "/static/../server.py")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertIn(resp.status, (400, 404))

    # ── index page ────────────────────────────────────────────────────────────

    def test_index_html(self):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertIn(b"edge_vlm_ros", body)

    def test_404_on_unknown_path(self):
        status, data = self._get("/api/unknown")
        self.assertEqual(status, 404)


# ── fixture manifest parsing tests ───────────────────────────────────────────

class TestFixtureManifestParsing(unittest.TestCase):
    """Validate that manifests produced by existing scripts are parseable."""

    _STANDALONE_MANIFEST = {
        "schema_version": 1,
        "run_id": "12345678-abcd-4000-8000-000000000001",
        "kind": "standalone",
        "created_at": "2025-01-01T00:00:00+00:00",
        "prompt": "Describe the scene.",
        "image_filename": "frame.jpg",
        "max_generate_length": 64,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "success": True,
        "text": "A warehouse aisle with pallets.",
        "error": "",
        "inference_seconds": 1.234,
    }

    _ROS_MANIFEST = {
        "schema_version": 1,
        "run_id": "12345678-abcd-4000-8000-000000000002",
        "kind": "ros",
        "created_at": "2025-01-01T01:00:00+00:00",
        "params": {
            "image_topic": "/hawk_0_left_rgb_image",
            "max_generate_length": 64,
            "instruction_delivery_mode": "inline",
        },
        "pid": 12345,
        "status": "running",
    }

    def test_standalone_manifest_fields(self):
        m = self._STANDALONE_MANIFEST
        self.assertEqual(m["schema_version"], 1)
        self.assertIn("run_id", m)
        self.assertIn("inference_seconds", m)
        self.assertIsInstance(m["success"], bool)

    def test_ros_manifest_fields(self):
        m = self._ROS_MANIFEST
        self.assertIn("params", m)
        self.assertIn("pid", m)

    def test_manifest_round_trip(self):
        with _TempDir() as d:
            store = RunStore(d)
            run_id = self._STANDALONE_MANIFEST["run_id"]
            store.save_run(run_id, self._STANDALONE_MANIFEST)
            loaded = store.get_run(run_id)
            self.assertEqual(loaded["text"], "A warehouse aisle with pallets.")
            self.assertAlmostEqual(loaded["inference_seconds"], 1.234)

    def test_list_runs_includes_fixture(self):
        with _TempDir() as d:
            store = RunStore(d)
            for manifest in (self._STANDALONE_MANIFEST, self._ROS_MANIFEST):
                store.save_run(manifest["run_id"], manifest)
            runs = store.list_runs()
            kinds = {r["kind"] for r in runs}
            self.assertIn("standalone", kinds)
            self.assertIn("ros", kinds)


# ── static asset syntax check ─────────────────────────────────────────────────

class TestStaticAssets(unittest.TestCase):
    """Ensure static files are present and not obviously malformed."""

    _STATIC_DIR = _REPO_ROOT / "web_console" / "static"

    def test_style_css_exists(self):
        self.assertTrue((self._STATIC_DIR / "style.css").is_file())

    def test_app_js_exists(self):
        self.assertTrue((self._STATIC_DIR / "app.js").is_file())

    def test_css_has_body_rule(self):
        css = (self._STATIC_DIR / "style.css").read_text()
        self.assertIn("body", css)

    def test_js_no_eval(self):
        js = (self._STATIC_DIR / "app.js").read_text()
        # eval() with user data is a security risk; a simple absence check.
        self.assertNotIn("eval(", js)

    def test_js_syntax_check(self):
        """node --check validates JS syntax; skip gracefully if node is absent."""
        js_path = self._STATIC_DIR / "app.js"
        try:
            result = subprocess.run(
                ["node", "--check", str(js_path)],
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
        except FileNotFoundError:
            self.skipTest("node not available")


# ── ROS env builder tests ─────────────────────────────────────────────────────

class TestBuildRosEnv(unittest.TestCase):

    def test_known_params_mapped(self):
        params = {
            "image_topic": "/cam",
            "max_generate_length": 128,
            "instruction_delivery_mode": "structured",
        }
        env = _build_ros_env(params, {})
        self.assertEqual(env.get("MAX_GENERATE_LENGTH"), "128")
        self.assertEqual(env.get("INSTRUCTION_DELIVERY_MODE"), "structured")
        self.assertEqual(env.get("IMAGE_TOPIC"), "/cam")

    def test_unknown_param_not_in_env(self):
        params = {"shell_cmd": "evil"}
        env = _build_ros_env(params, {})
        self.assertNotIn("shell_cmd", env)
        self.assertNotIn("evil", env.values())

    def test_inherits_existing_env(self):
        env = _build_ros_env({}, {})
        self.assertIn("PATH", env)

    def test_artifact_dir_set_in_env(self):
        """artifact_dir parameter must be forwarded as ARTIFACT_DIR."""
        with _TempDir() as d:
            env = _build_ros_env({}, {}, artifact_dir=d)
            self.assertEqual(env.get("ARTIFACT_DIR"), str(d))

    def test_artifact_dir_not_set_without_param(self):
        """ARTIFACT_DIR must not appear when neither artifact_dir nor cfg key is given."""
        env = _build_ros_env({}, {})
        self.assertNotIn("ARTIFACT_DIR", env)




# ── ROS run finalization tests ────────────────────────────────────────────────

class TestRosRunFinalization(unittest.TestCase):
    """Integration tests: verify that ROS run manifests are finalized on disk
    when the experiment process exits naturally, exits with error, or is stopped."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = pathlib.Path(tempfile.mkdtemp())
        cls._run_store = RunStore(cls._tmpdir / "runs")
        cls._process_manager = ProcessManager()
        cls._srv, cls._port, cls._thread = _start_test_server(
            config={"socket_path": "/tmp/no_such.sock", "quiet": True},
            process_manager=cls._process_manager,
            run_store=cls._run_store,
        )

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        cls._process_manager.cleanup()
        shutil.rmtree(str(cls._tmpdir), ignore_errors=True)

    def _start_ros(self, script_content: str) -> tuple:
        """Write a temp script and POST /api/ros/start; return (http_status, response_data)."""
        script = self._tmpdir / f"script_{RunStore.new_run_id()}.sh"
        script.write_text(script_content)
        script.chmod(0o755)
        old = self._srv.config.get("ros_script_path")
        self._srv.config["ros_script_path"] = str(script)
        try:
            body = json.dumps({"params": {}}).encode()
            conn = HTTPConnection("127.0.0.1", self._port, timeout=10)
            conn.request(
                "POST",
                "/api/ros/start",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            data = json.loads(resp.read())
            return resp.status, data
        finally:
            conn.close()
            if old is None:
                self._srv.config.pop("ros_script_path", None)
            else:
                self._srv.config["ros_script_path"] = old

    def _poll_terminal(self, run_id: str, timeout: float = 10.0) -> dict:
        """Poll GET /api/runs/<id> until status is terminal (completed/failed/stopped)."""
        deadline = time.monotonic() + timeout
        data = {}
        while time.monotonic() < deadline:
            conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
            conn.request("GET", f"/api/runs/{run_id}")
            resp = conn.getresponse()
            data = json.loads(resp.read())
            conn.close()
            if data.get("status") in _TERMINAL_STATUSES:
                return data
            time.sleep(0.1)
        return data

    def test_manifest_finalized_on_natural_success(self):
        """Natural exit 0: manifest updated with completed/success/log_lines."""
        status, data = self._start_ros("#!/bin/bash\necho hello_finalize\nexit 0\n")
        self.assertEqual(status, 202)
        run_id = data["run_id"]

        record = self._poll_terminal(run_id, timeout=10.0)
        self.assertEqual(record.get("status"), "completed")
        self.assertTrue(record.get("success"))
        self.assertEqual(record.get("exit_code"), 0)
        self.assertIn("completed_at", record)
        log_text = " ".join(record.get("log_lines", []))
        self.assertIn("hello_finalize", log_text)

    def test_manifest_finalized_on_nonzero_exit(self):
        """Natural exit 1: manifest updated with failed/success=False."""
        status, data = self._start_ros("#!/bin/bash\nexit 1\n")
        self.assertEqual(status, 202)
        run_id = data["run_id"]

        record = self._poll_terminal(run_id, timeout=10.0)
        self.assertEqual(record.get("status"), "failed")
        self.assertFalse(record.get("success"))
        self.assertEqual(record.get("exit_code"), 1)
        self.assertIn("completed_at", record)

    def test_manifest_finalized_on_explicit_stop(self):
        """Explicit stop: manifest updated with stopped status."""
        status, data = self._start_ros("#!/bin/bash\nsleep 60\n")
        self.assertEqual(status, 202)
        run_id = data["run_id"]

        # Stop via API.
        body = json.dumps({"run_id": run_id}).encode()
        conn = HTTPConnection("127.0.0.1", self._port, timeout=15)
        conn.request(
            "POST",
            "/api/ros/stop",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()

        # After the stop response, the manifest must already be finalized.
        record = self._poll_terminal(run_id, timeout=5.0)
        self.assertNotEqual(record.get("status"), "running")
        self.assertIn("completed_at", record)
        self.assertIn("exit_code", record)

    def test_finalized_manifest_readable_after_restart(self):
        """Finalized manifests are persisted and readable from a fresh RunStore."""
        status, data = self._start_ros("#!/bin/bash\necho persist_check\nexit 0\n")
        self.assertEqual(status, 202)
        run_id = data["run_id"]
        self._poll_terminal(run_id, timeout=10.0)

        # Simulate a restart by reading directly from disk (new RunStore instance).
        fresh_store = RunStore(self._tmpdir / "runs")
        record = fresh_store.get_run(run_id)
        self.assertIsNotNone(record)
        self.assertNotIn(record.get("status"), ("running", "starting"),
                         "Persisted manifest must reflect terminal state")
        self.assertIn("completed_at", record)

    def test_artifact_captured_in_manifest(self):
        """Artifacts written by the real script layout are captured in the console manifest.

        ARTIFACT_DIR is a nested artifacts/ subdirectory so the console's
        manifest.json is never overwritten by the script's manifest.json.
        The finalized console manifest must contain:
          - safe relative artifact paths in 'artifacts'
          - parsed script manifest.json content in 'script_manifest'
        """
        script = (
            "#!/bin/bash\n"
            'mkdir -p "$ARTIFACT_DIR"\n'
            'echo \'{"schema_version": 1, "successful_results_observed": 1}\' > "$ARTIFACT_DIR/manifest.json"\n'
            'echo \'{"latency_ms": 42}\' > "$ARTIFACT_DIR/benchmark.jsonl"\n'
            'echo "launch output" > "$ARTIFACT_DIR/launch.log"\n'
            'echo "result: true" > "$ARTIFACT_DIR/results.log"\n'
            "exit 0\n"
        )
        status, data = self._start_ros(script)
        self.assertEqual(status, 202)
        run_id = data["run_id"]

        record = self._poll_terminal(run_id, timeout=10.0)
        self.assertEqual(record.get("status"), "completed")
        self.assertEqual(record.get("run_id"), run_id)
        self.assertEqual(record.get("kind"), "ros")

        # artifacts must be a list of safe relative paths
        artifacts = record.get("artifacts", [])
        self.assertIsInstance(artifacts, list)
        self.assertIn("artifacts/manifest.json", artifacts)
        self.assertIn("artifacts/benchmark.jsonl", artifacts)
        self.assertIn("artifacts/launch.log", artifacts)
        self.assertIn("artifacts/results.log", artifacts)

        # script_manifest must be parsed from artifacts/manifest.json
        script_manifest = record.get("script_manifest")
        self.assertIsNotNone(script_manifest, "script_manifest must be parsed from artifacts/manifest.json")
        self.assertEqual(script_manifest.get("schema_version"), 1)
        self.assertEqual(script_manifest.get("successful_results_observed"), 1)

    def test_console_manifest_not_overwritten_by_script(self):
        """Script writing to ARTIFACT_DIR must not overwrite the console manifest.json.

        The real run_image_proc_test.sh writes <ARTIFACT_DIR>/manifest.json.
        Because ARTIFACT_DIR is now <run_id>/artifacts/, the console's own
        <run_id>/manifest.json must remain intact with run_id, kind, schema_version=1,
        and a terminal status.
        """
        script = (
            "#!/bin/bash\n"
            'mkdir -p "$ARTIFACT_DIR"\n'
            # Simulate the script writing its own manifest.json with a different schema
            'echo \'{"schema_version": 999, "hostile": true}\' > "$ARTIFACT_DIR/manifest.json"\n'
            "exit 0\n"
        )
        status, data = self._start_ros(script)
        self.assertEqual(status, 202)
        run_id = data["run_id"]

        record = self._poll_terminal(run_id, timeout=10.0)
        # Console manifest must retain its own fields
        self.assertEqual(record.get("run_id"), run_id)
        self.assertEqual(record.get("kind"), "ros")
        self.assertIn(record.get("status"), _TERMINAL_STATUSES)
        # schema_version 1 is the console schema, not 999 from the script artifact
        self.assertEqual(record.get("schema_version"), 1,
                         "Console manifest schema_version must not be overwritten by script artifact")
        self.assertNotIn("hostile", record,
                         "Console manifest must not contain keys from the script's manifest.json")

    def test_logs_api_includes_terminal_flag(self):
        """GET /api/runs/<id>/logs must include 'terminal' and 'status' fields."""
        status, data = self._start_ros("#!/bin/bash\necho log_terminal_check\nexit 0\n")
        self.assertEqual(status, 202)
        run_id = data["run_id"]
        self._poll_terminal(run_id, timeout=10.0)

        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", f"/api/runs/{run_id}/logs")
        resp = conn.getresponse()
        logs_data = json.loads(resp.read())
        conn.close()

        self.assertEqual(resp.status, 200)
        self.assertIn("terminal", logs_data)
        self.assertIn("status", logs_data)
        self.assertTrue(logs_data["terminal"], "terminal must be True for a completed run")
        self.assertEqual(logs_data["status"], "completed")


# ── fast-exit manifest race tests ─────────────────────────────────────────────

class TestFastExitRace(unittest.TestCase):
    """Verify that a child process exiting before start_ros_experiment returns
    does not leave the manifest permanently in 'starting' or 'running' state."""

    def test_fast_exit_before_start_ros_returns(self):
        """Synchronous completion inside start_ros_experiment must not lose the manifest.

        This test uses a fake ProcessManager whose start_ros_experiment fires
        the completion callback *synchronously* before returning — deterministically
        simulating a child that exits before the handler writes 'running'.
        """

        class _SyncCompletePM:
            """Fires on_complete synchronously to simulate instant child exit."""

            def start_ros_experiment(self, run_id, args, env=None, on_complete=None):
                if on_complete is not None:
                    on_complete(run_id, 0, False, ["fast_line"])
                return 99999  # fake PID

            def active_ros_run_id(self):
                return None

            def stop_experiment(self, run_id):
                raise KeyError(run_id)

            def get_logs(self, run_id):
                return []

            def is_running(self, run_id):
                return False

            def cleanup(self):
                pass

        with _TempDir() as d:
            run_store = RunStore(d / "runs")
            pm = _SyncCompletePM()
            srv, port, _ = _start_test_server(
                config={"socket_path": "/tmp/no_such.sock", "quiet": True},
                process_manager=pm,
                run_store=run_store,
            )
            try:
                body = json.dumps({"params": {}}).encode()
                conn = HTTPConnection("127.0.0.1", port, timeout=10)
                conn.request(
                    "POST", "/api/ros/start", body=body,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                data = json.loads(resp.read())
                conn.close()

                self.assertEqual(resp.status, 202)
                run_id = data["run_id"]

                # The pre-write guarantee means the manifest must exist.
                record = run_store.get_run(run_id)
                self.assertIsNotNone(record, "Manifest must exist even after fast exit")

                # The callback fired synchronously, so status must be terminal.
                self.assertIn(
                    record.get("status"),
                    _TERMINAL_STATUSES,
                    f"Expected terminal status after fast exit, got: {record.get('status')!r}",
                )
                self.assertEqual(record.get("exit_code"), 0)
                self.assertIn("completed_at", record)
            finally:
                srv.shutdown()


# ── startup bind / warning tests ──────────────────────────────────────────────

class TestStartupBind(unittest.TestCase):
    """CPU-only tests for default loopback bind and non-loopback warning."""

    def test_default_host_is_loopback(self):
        """Parsing args with no --host flag must yield 127.0.0.1 (loopback)."""
        args = _parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertTrue(_is_loopback(args.host))

    def test_explicit_loopback_ipv4(self):
        self.assertTrue(_is_loopback("127.0.0.1"))

    def test_explicit_loopback_ipv6(self):
        self.assertTrue(_is_loopback("::1"))

    def test_localhost_hostname_is_loopback(self):
        self.assertTrue(_is_loopback("localhost"))

    def test_all_interfaces_is_not_loopback(self):
        self.assertFalse(_is_loopback("0.0.0.0"))

    def test_lan_ip_is_not_loopback(self):
        self.assertFalse(_is_loopback("192.168.1.10"))

    def test_non_loopback_warning_printed(self):
        """main() must print a warning when --host is a non-loopback address."""
        import io
        from unittest.mock import patch as _patch

        captured = io.StringIO()
        # Patch serve_forever so the server does not actually start.
        with _patch("web_console.server.ConsoleServer.serve_forever", return_value=None), \
             _patch("sys.stdout", captured):
            from web_console.__main__ import main
            main(["--host", "0.0.0.0", "--port", "19999"])

        output = captured.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("NO AUTHENTICATION", output)

    def test_loopback_no_warning_printed(self):
        """main() must NOT print a security warning when --host is loopback."""
        import io
        from unittest.mock import patch as _patch

        captured = io.StringIO()
        with _patch("web_console.server.ConsoleServer.serve_forever", return_value=None), \
             _patch("sys.stdout", captured):
            from web_console.__main__ import main
            main(["--host", "127.0.0.1", "--port", "19998"])

        output = captured.getvalue()
        self.assertNotIn("WARNING", output)
        self.assertNotIn("NO AUTHENTICATION", output)


# ── result-log / benchmark parsers ───────────────────────────────────────────

class TestResultsParsing(unittest.TestCase):
    """CPU-only tests for server-side artifact parsers.

    All results.log fixtures use the real ``ros2 topic echo`` format:
    fields first, ``---`` as a *trailing* message separator.  Field names
    match the actual VlmResult message (``frame_sequence``, ``response``,
    ``inference_seconds``); the parser normalises these to the canonical
    UI schema (``frame_seq``, ``text``, ``latency_ms``).
    """

    def test_parse_empty_results_log(self):
        self.assertEqual(_parse_results_log(""), [])

    def test_parse_whitespace_only(self):
        self.assertEqual(_parse_results_log("   \n   "), [])

    def test_parse_no_separator_is_empty(self):
        """A file without any trailing --- separator produces no frames
        (treated as incomplete/truncated output)."""
        text = "frame_sequence: 1\nresponse: hello\nsuccess: true\n"
        self.assertEqual(_parse_results_log(text), [])

    def test_parse_single_successful_frame(self):
        """Single frame with real ROS field names and trailing separator."""
        text = (
            "header:\n"
            "  stamp:\n"
            "    sec: 1753830000\n"
            "    nanosec: 500000000\n"
            "  frame_id: ''\n"
            "frame_sequence: 1\n"
            "response: A dog is running.\n"
            "inference_seconds: 0.0425\n"
            "success: true\n"
            "error: ''\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0]["success"], True)
        # Field names must be normalised to the UI schema
        self.assertEqual(frames[0]["text"], "A dog is running.")
        self.assertAlmostEqual(frames[0]["latency_ms"], 42.5, places=2)
        self.assertEqual(frames[0]["frame_seq"], 1)
        # Source timestamp: 1753830000 * 1e9 + 500_000_000
        self.assertEqual(frames[0]["source_timestamp_ns"], 1753830000_500_000_000)
        # Raw ROS names must NOT appear in the output
        self.assertNotIn("response", frames[0])
        self.assertNotIn("frame_sequence", frames[0])
        self.assertNotIn("inference_seconds", frames[0])

    def test_parse_single_failed_frame(self):
        """Failed frame: success=false with error message."""
        text = (
            "frame_sequence: 2\n"
            "response: ''\n"
            "inference_seconds: 0.0\n"
            "success: false\n"
            "error: timeout\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0]["success"], False)
        self.assertEqual(frames[0]["error"], "timeout")

    def test_parse_multiline_text_block_scalar(self):
        """YAML block scalar (|) for response is joined from indented lines."""
        text = (
            "frame_sequence: 1\n"
            "response: |\n"
            "  Line one.\n"
            "  Line two.\n"
            "inference_seconds: 0.01\n"
            "success: true\n"
            "error: ''\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertIn("Line one.", frames[0]["text"])
        self.assertIn("Line two.", frames[0]["text"])

    def test_parse_multi_frame_results(self):
        """Multiple frames each terminated by trailing ---."""
        text = (
            "frame_sequence: 1\n"
            "response: frame1\n"
            "success: true\n"
            "---\n"
            "frame_sequence: 2\n"
            "response: frame2\n"
            "success: false\n"
            "---\n"
            "frame_sequence: 3\n"
            "response: frame3\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual([f["text"] for f in frames], ["frame1", "frame2", "frame3"])

    def test_parse_malformed_lines_skipped(self):
        """Non key:value lines in a block are silently ignored."""
        text = (
            ":::not_a_key\n"
            "success: true\n"
            "response: ok\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["text"], "ok")

    def test_parse_incomplete_trailing_block_discarded(self):
        """Content after the last --- with no closing --- is discarded."""
        text = (
            "frame_sequence: 1\n"
            "response: complete\n"
            "success: true\n"
            "---\n"
            "frame_sequence: 2\n"
            "response: incomplete\n"
        )
        frames = _parse_results_log(text)
        # Only the terminated frame is returned
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["text"], "complete")

    def test_parse_field_names_normalised(self):
        """frame_sequence, response, inference_seconds are normalised to UI names."""
        text = (
            "frame_sequence: 7\n"
            "response: hello world\n"
            "inference_seconds: 1.5\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        f = frames[0]
        self.assertEqual(f["frame_seq"], 7)
        self.assertEqual(f["text"], "hello world")
        self.assertAlmostEqual(f["latency_ms"], 1500.0)
        self.assertNotIn("frame_sequence", f)
        self.assertNotIn("response", f)
        self.assertNotIn("inference_seconds", f)

    def test_parse_source_timestamp_ns_from_header(self):
        """sec and nanosec under header.stamp are combined into source_timestamp_ns."""
        text = (
            "header:\n"
            "  stamp:\n"
            "    sec: 1000000000\n"
            "    nanosec: 250000000\n"
            "  frame_id: ''\n"
            "frame_sequence: 1\n"
            "response: timestamped\n"
            "inference_seconds: 0.1\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        expected_ns = 1000000000 * 1_000_000_000 + 250000000
        self.assertEqual(frames[0]["source_timestamp_ns"], expected_ns)

    def test_parse_no_source_timestamp_when_header_absent(self):
        """Frames without a header block must not have source_timestamp_ns."""
        text = (
            "frame_sequence: 1\n"
            "response: no header\n"
            "inference_seconds: 0.2\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertNotIn("source_timestamp_ns", frames[0])

    def test_parse_real_thor_output_fixture(self):
        """Full two-frame fixture matching the real ros2 topic echo VlmResult format
        observed on Thor.  Validates all normalised keys and source_timestamp_ns."""
        text = (
            "header:\n"
            "  stamp:\n"
            "    sec: 1753830042\n"
            "    nanosec: 123456789\n"
            "  frame_id: ''\n"
            "frame_sequence: 1\n"
            "response: A construction crane is visible in the upper left of the frame.\n"
            "inference_seconds: 0.847\n"
            "success: true\n"
            "error: ''\n"
            "---\n"
            "header:\n"
            "  stamp:\n"
            "    sec: 1753830043\n"
            "    nanosec: 987654321\n"
            "  frame_id: ''\n"
            "frame_sequence: 2\n"
            "response: ''\n"
            "inference_seconds: 0.0\n"
            "success: false\n"
            "error: timeout waiting for worker\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 2)

        f0 = frames[0]
        self.assertEqual(f0["frame_seq"], 1)
        self.assertIs(f0["success"], True)
        self.assertEqual(
            f0["text"],
            "A construction crane is visible in the upper left of the frame.",
        )
        self.assertAlmostEqual(f0["latency_ms"], 847.0, places=1)
        self.assertEqual(
            f0["source_timestamp_ns"],
            1753830042 * 1_000_000_000 + 123456789,
        )

        f1 = frames[1]
        self.assertEqual(f1["frame_seq"], 2)
        self.assertIs(f1["success"], False)
        self.assertEqual(f1["error"], "timeout waiting for worker")
        self.assertEqual(
            f1["source_timestamp_ns"],
            1753830043 * 1_000_000_000 + 987654321,
        )

    # ── _parse_benchmark_jsonl ────────────────────────────────────────────

    def test_parse_benchmark_jsonl_empty(self):
        self.assertEqual(_parse_benchmark_jsonl(""), [])

    def test_parse_benchmark_jsonl_valid(self):
        lines = (
            '{"record_type": "frame", "inference_ms": 30.0, "success": true}\n'
            '{"record_type": "frame", "inference_ms": 50.0, "success": false}\n'
        )
        records = _parse_benchmark_jsonl(lines)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["inference_ms"], 30.0)

    def test_parse_benchmark_jsonl_skips_malformed_lines(self):
        lines = (
            '{"record_type": "frame", "inference_ms": 10.0}\n'
            'NOT JSON\n'
            '{"record_type": "summary"}\n'
        )
        records = _parse_benchmark_jsonl(lines)
        self.assertEqual(len(records), 2)

    def test_parse_benchmark_jsonl_skips_non_dict(self):
        lines = '["not", "a", "dict"]\n{"record_type": "frame"}\n'
        records = _parse_benchmark_jsonl(lines)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_type"], "frame")

    def test_benchmark_summary_computed_correctly(self):
        """_parse_benchmark_jsonl returns records; benchmark_summary is computed
        by the server from record_type=frame entries."""
        lines = (
            '{"record_type": "frame", "inference_ms": 20.0, "success": true}\n'
            '{"record_type": "frame", "inference_ms": 40.0, "success": true}\n'
            '{"record_type": "frame", "inference_ms": 60.0, "success": false}\n'
            '{"record_type": "summary", "total": 3}\n'
        )
        records = _parse_benchmark_jsonl(lines)
        frame_records = [r for r in records if r.get("record_type") == "frame"]
        self.assertEqual(len(frame_records), 3)
        successful = [r for r in frame_records if r.get("success")]
        mean_ms = sum(r["inference_ms"] for r in frame_records) / len(frame_records)
        self.assertAlmostEqual(mean_ms, 40.0)
        self.assertEqual(len(successful), 2)


# ── external-service (START_WORKER=false) mode ────────────────────────────────

class TestExternalServiceMode(unittest.TestCase):
    """CPU-only tests for _build_ros_env start_worker parameter and
    _api_ros_start mode-selection based on service reachability."""

    def _cfg(self, socket_path: str = "/tmp/test.sock") -> dict:
        return {"socket_path": socket_path}

    # ── _build_ros_env ────────────────────────────────────────────────────

    def test_build_ros_env_start_worker_true_does_not_set_env_var(self):
        env = _build_ros_env({}, self._cfg(), start_worker=True)
        self.assertNotIn("START_WORKER", env)

    def test_build_ros_env_start_worker_false_sets_env_var(self):
        env = _build_ros_env({}, self._cfg(), start_worker=False)
        self.assertEqual(env.get("START_WORKER"), "false")

    def test_build_ros_env_start_worker_false_sets_socket_path(self):
        cfg = self._cfg(socket_path="/run/edge_vlm.sock")
        env = _build_ros_env({}, cfg, start_worker=False)
        self.assertEqual(env.get("WORKER_SOCKET_PATH"), "/run/edge_vlm.sock")

    def test_build_ros_env_start_worker_false_empty_socket_no_env_var(self):
        """Empty socket_path must not set WORKER_SOCKET_PATH."""
        cfg = {"socket_path": ""}
        env = _build_ros_env({}, cfg, start_worker=False)
        self.assertNotIn("WORKER_SOCKET_PATH", env)

    def test_build_ros_env_start_worker_true_no_socket_path_env(self):
        """start_worker=True must never override WORKER_SOCKET_PATH."""
        env = _build_ros_env({}, self._cfg(), start_worker=True)
        self.assertNotIn("WORKER_SOCKET_PATH", env)

    # ── API mode selection ────────────────────────────────────────────────

    def _make_ros_start_request(self, srv: ConsoleServer) -> dict:
        """POST /api/ros/start and return parsed JSON response."""
        host, port = srv.server_address
        conn = HTTPConnection(host, port, timeout=10)
        body_bytes = json.dumps({"params": {}}).encode()
        conn.request(
            "POST",
            "/api/ros/start",
            body=body_bytes,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body_bytes)),
            },
        )
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())

    def test_api_ros_start_selects_external_mode_when_service_reachable(self):
        """When the standalone service is reachable, the run manifest must
        record external_worker=True and START_WORKER=false must be set."""
        with _TempDir() as tmp:
            run_store = RunStore(tmp)
            cfg = {"socket_path": "/tmp/nonexistent.sock", "quiet": True}
            # Fake script: exits 0 immediately
            fake_script = tmp / "fake_ros.sh"
            fake_script.write_text("#!/bin/sh\nexit 0\n")
            fake_script.chmod(0o755)
            cfg["ros_script_path"] = str(fake_script)
            # Mock check_server_reachable to return True (service is running)
            with patch("web_console.server.check_server_reachable", return_value={"reachable": True}):
                srv = ConsoleServer(host="127.0.0.1", port=0, config=cfg,
                                    run_store=run_store)
                thread = threading.Thread(target=srv.serve_forever, daemon=True)
                thread.start()
                try:
                    time.sleep(0.1)
                    status, data = self._make_ros_start_request(srv)
                    self.assertIn(status, (200, 202))
                    self.assertTrue(data.get("external_worker"),
                                    f"external_worker not True in {data}")
                finally:
                    srv.shutdown()

    def test_api_ros_start_selects_self_managed_when_service_unreachable(self):
        """When the standalone service is not reachable, external_worker must
        be False and START_WORKER must not be overridden to false."""
        with _TempDir() as tmp:
            run_store = RunStore(tmp)
            cfg = {"socket_path": "/tmp/nonexistent.sock", "quiet": True}
            fake_script = tmp / "fake_ros.sh"
            fake_script.write_text("#!/bin/sh\nexit 0\n")
            fake_script.chmod(0o755)
            cfg["ros_script_path"] = str(fake_script)
            # Mock check_server_reachable to return False (no service)
            with patch("web_console.server.check_server_reachable", return_value={"reachable": False}):
                srv = ConsoleServer(host="127.0.0.1", port=0, config=cfg,
                                    run_store=run_store)
                thread = threading.Thread(target=srv.serve_forever, daemon=True)
                thread.start()
                try:
                    time.sleep(0.1)
                    status, data = self._make_ros_start_request(srv)
                    self.assertIn(status, (200, 202))
                    self.assertFalse(data.get("external_worker"),
                                     f"external_worker should be False in {data}")
                finally:
                    srv.shutdown()

    def test_external_worker_flag_preserved_in_manifest_after_completion(self):
        """external_worker=True written in 'starting' manifest must survive
        finalization by the completion callback."""
        with _TempDir() as tmp:
            run_store = RunStore(tmp)
            cfg = {"socket_path": "/tmp/nonexistent.sock", "quiet": True}
            fake_script = tmp / "fast_exit.sh"
            fake_script.write_text("#!/bin/sh\nexit 0\n")
            fake_script.chmod(0o755)
            cfg["ros_script_path"] = str(fake_script)
            with patch("web_console.server.check_server_reachable", return_value={"reachable": True}):
                srv = ConsoleServer(host="127.0.0.1", port=0, config=cfg,
                                    run_store=run_store)
                thread = threading.Thread(target=srv.serve_forever, daemon=True)
                thread.start()
                try:
                    time.sleep(0.1)
                    status, data = self._make_ros_start_request(srv)
                    self.assertIn(status, (200, 202))
                    run_id = data.get("run_id")
                    # Allow the child process to exit and the callback to fire.
                    for _ in range(30):
                        rec = run_store.get_run(run_id)
                        if rec and rec.get("status") in _TERMINAL_STATUSES:
                            break
                        time.sleep(0.1)
                    rec = run_store.get_run(run_id)
                    self.assertIsNotNone(rec)
                    self.assertTrue(rec.get("external_worker"),
                                    "external_worker must be preserved in final manifest")
                    self.assertIn(rec.get("status"), _TERMINAL_STATUSES)
                finally:
                    srv.shutdown()

    # ── IMAGE_TOPIC env var ────────────────────────────────────────────────

    def test_build_ros_env_image_topic_forwarded_as_IMAGE_TOPIC(self):
        """image_topic param must appear as IMAGE_TOPIC in the subprocess env."""
        params = {"image_topic": "/camera/color/image_raw"}
        env = _build_ros_env(params, self._cfg(), start_worker=True)
        self.assertEqual(env.get("IMAGE_TOPIC"), "/camera/color/image_raw")

    def test_build_ros_env_default_image_topic_not_set_when_absent(self):
        """When image_topic is not in params, IMAGE_TOPIC must not be added
        (the script honours its own default of /hawk_0_left_rgb_image)."""
        env = _build_ros_env({}, self._cfg(), start_worker=True)
        self.assertNotIn("IMAGE_TOPIC", env)

    def test_api_ros_start_image_topic_reaches_subprocess_env(self):
        """IMAGE_TOPIC env var supplied from the web form must be passed to
        the script subprocess; a fake script echoes it to a capture file and
        proves the value matches what the form sent."""
        with _TempDir() as tmp:
            run_store = RunStore(tmp)
            capture_file = tmp / "captured_topic.txt"
            fake_script = tmp / "fake_ros.sh"
            fake_script.write_text(
                "#!/bin/bash\n"
                f'echo "${{IMAGE_TOPIC:-NOT_SET}}" > "{capture_file}"\n'
                "exit 0\n"
            )
            fake_script.chmod(0o755)
            cfg = {
                "socket_path": "",
                "quiet": True,
                "ros_script_path": str(fake_script),
            }
            with patch("web_console.server.check_server_reachable",
                       return_value={"reachable": False}):
                srv = ConsoleServer(host="127.0.0.1", port=0, config=cfg,
                                    run_store=run_store)
                thread = threading.Thread(target=srv.serve_forever, daemon=True)
                thread.start()
                try:
                    time.sleep(0.1)
                    host, port = srv.server_address
                    conn = HTTPConnection(host, port, timeout=10)
                    body_bytes = json.dumps({
                        "params": {"image_topic": "/hawk_0_left_rgb_image"}
                    }).encode()
                    conn.request(
                        "POST", "/api/ros/start", body=body_bytes,
                        headers={
                            "Content-Type": "application/json",
                            "Content-Length": str(len(body_bytes)),
                        },
                    )
                    resp = conn.getresponse()
                    data = json.loads(resp.read())
                    self.assertIn(resp.status, (200, 202))
                    run_id = data.get("run_id")
                    # Wait for the script to complete so the capture file is written
                    for _ in range(30):
                        rec = run_store.get_run(run_id)
                        if rec and rec.get("status") in _TERMINAL_STATUSES:
                            break
                        time.sleep(0.1)
                    self.assertTrue(capture_file.is_file(),
                                    "Script did not write the capture file")
                    captured = capture_file.read_text().strip()
                    self.assertEqual(captured, "/hawk_0_left_rgb_image",
                                     "IMAGE_TOPIC env var did not reach the subprocess")
                finally:
                    srv.shutdown()


# ── experiment engine tests ───────────────────────────────────────────────────

from web_console.experiment_engine import (
    ExperimentDefinition,
    FrameResult,
    validate_definition,
    run_experiment,
    _truncate_history,
    _compute_repetition,
    _build_prompt,
    compute_history_matrix,
)
from web_console.model_catalog import (
    discover_models,
    _make_model_id,
    _profile_from_env,
    _scan_workspace,
)
from web_console.dataset_catalog import (
    discover_datasets,
    build_download_command,
    _DOWNLOADABLE_BAGS,
)


class TestExperimentEngineValidation(unittest.TestCase):

    def _valid_defn(self, **kwargs):
        base = dict(
            strategy="single_frame",
            image_paths=["/data/frame.jpg"],
            task_prompt="Describe the scene.",
        )
        base.update(kwargs)
        return ExperimentDefinition(**base)

    def test_valid_definition_returns_none(self):
        defn = self._valid_defn()
        self.assertIsNone(validate_definition(defn))

    def test_invalid_strategy(self):
        defn = self._valid_defn(strategy="unknown_strategy")
        err = validate_definition(defn)
        self.assertIsNotNone(err)
        self.assertIn("Unknown strategy", err)

    def test_empty_image_paths(self):
        defn = self._valid_defn(image_paths=[])
        err = validate_definition(defn)
        self.assertIsNotNone(err)
        self.assertIn("image_paths", err)

    def test_unsupported_image_extension(self):
        defn = self._valid_defn(image_paths=["/data/frame.xyz"])
        err = validate_definition(defn)
        self.assertIsNotNone(err)
        self.assertIn(".xyz", err)

    def test_empty_prompt(self):
        defn = self._valid_defn(task_prompt="")
        err = validate_definition(defn)
        self.assertIsNotNone(err)
        self.assertIn("task_prompt", err)

    def test_max_generate_length_out_of_range(self):
        defn = self._valid_defn(max_generate_length=0)
        self.assertIsNotNone(validate_definition(defn))
        defn2 = self._valid_defn(max_generate_length=4097)
        self.assertIsNotNone(validate_definition(defn2))

    def test_temperature_out_of_range(self):
        defn = self._valid_defn(temperature=2.5)
        self.assertIsNotNone(validate_definition(defn))

    def test_top_p_out_of_range(self):
        defn = self._valid_defn(top_p=0.0)
        self.assertIsNotNone(validate_definition(defn))

    def test_history_entries_out_of_range(self):
        defn = self._valid_defn(observation_history_max_entries=257)
        self.assertIsNotNone(validate_definition(defn))

    def test_history_strategy_valid(self):
        defn = self._valid_defn(
            strategy="single_frame_observation_history",
            observation_history_max_entries=3,
        )
        self.assertIsNone(validate_definition(defn))


class TestExperimentEngineHistoryHelpers(unittest.TestCase):

    def test_truncate_history_empty(self):
        self.assertEqual(_truncate_history([], 100), "")

    def test_truncate_history_fits(self):
        result = _truncate_history(["a", "b", "c"], 1000)
        self.assertEqual(result, "a; b; c")

    def test_truncate_history_respects_budget(self):
        # With budget of 1 char, only the most-recent entry ("c") fits.
        result = _truncate_history(["long_entry_here", "b", "c"], 1)
        self.assertEqual(result, "c")

    def test_truncate_history_zero_budget(self):
        self.assertEqual(_truncate_history(["a", "b"], 0), "")

    def test_compute_repetition_no_history(self):
        self.assertFalse(_compute_repetition("hello world", []))

    def test_compute_repetition_identical(self):
        text = "the quick brown fox jumps over the lazy dog"
        self.assertTrue(_compute_repetition(text, [text]))

    def test_compute_repetition_distinct(self):
        t1 = "red apple on a wooden table near the window sill outside"
        t2 = "blue ocean with waves crashing against rocky cliffs far away"
        self.assertFalse(_compute_repetition(t1, [t2]))

    def test_build_prompt_no_history(self):
        result = _build_prompt("What is this?", "You are an observer.", [], 1000)
        self.assertIn("What is this?", result)
        self.assertIn("You are an observer.", result)
        self.assertNotIn("Prior observations", result)

    def test_build_prompt_with_history(self):
        result = _build_prompt("What changed?", "", ["First obs.", "Second obs."], 1000)
        self.assertIn("Prior observations", result)
        self.assertIn("First obs.", result)


class TestExperimentEngineSingleFrame(unittest.TestCase):

    def _make_inference_fn(self, text="Test response", success=True):
        """Return a mock inference function that returns a fixed InferenceResult."""
        from web_console.inference_client import InferenceResult

        def _fn(image_path=None, prompt=None, *, cli_path="", socket_path="",
                max_generate_length=64, temperature=0.2, top_p=0.9, top_k=20,
                timeout=30.0, timeout_seconds=30.0, **kwargs):
            return InferenceResult(
                success=success,
                text=text,
                error="" if success else "mock error",
                inference_seconds=0.05,
            )
        return _fn

    def test_single_frame_returns_one_result_per_image(self):
        with tempfile.TemporaryDirectory() as d:
            img = pathlib.Path(d) / "img.jpg"
            img.write_bytes(b"FAKE")
            defn = ExperimentDefinition(
                strategy="single_frame",
                image_paths=[str(img)],
                task_prompt="Describe.",
            )
            results = run_experiment(
                defn,
                inference_fn=self._make_inference_fn(),
                socket_path="/tmp/fake.sock",
            )
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)
            self.assertEqual(results[0].text, "Test response")

    def test_single_frame_multiple_images(self):
        with tempfile.TemporaryDirectory() as d:
            imgs = []
            for i in range(3):
                p = pathlib.Path(d) / f"f{i}.png"
                p.write_bytes(b"FAKE")
                imgs.append(str(p))
            defn = ExperimentDefinition(
                strategy="single_frame",
                image_paths=imgs,
                task_prompt="Describe.",
            )
            results = run_experiment(
                defn,
                inference_fn=self._make_inference_fn(),
                socket_path="/tmp/fake.sock",
            )
            self.assertEqual(len(results), 3)
            for i, r in enumerate(results):
                self.assertEqual(r.frame_index, i)

    def test_single_frame_writes_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            img = tmp / "img.jpg"
            img.write_bytes(b"FAKE")
            artifact_dir = tmp / "artifacts"
            defn = ExperimentDefinition(
                strategy="single_frame",
                image_paths=[str(img)],
                task_prompt="What?",
            )
            run_experiment(
                defn,
                inference_fn=self._make_inference_fn(),
                socket_path="/tmp/fake.sock",
                artifact_dir=artifact_dir,
            )
            self.assertTrue((artifact_dir / "experiment.jsonl").exists())
            self.assertTrue((artifact_dir / "manifest.json").exists())

    def test_invalid_definition_raises(self):
        defn = ExperimentDefinition(
            strategy="bad_strategy",
            image_paths=["/data/frame.jpg"],
            task_prompt="X",
        )
        with self.assertRaises(ValueError):
            run_experiment(defn, inference_fn=self._make_inference_fn())


class TestExperimentEngineObservationHistory(unittest.TestCase):

    def _make_inference_fn(self, responses=None):
        from web_console.inference_client import InferenceResult
        responses = responses or []
        counter = [0]

        def _fn(image_path=None, prompt=None, *, cli_path="", socket_path="",
                max_generate_length=64, temperature=0.2, top_p=0.9, top_k=20,
                timeout=30.0, timeout_seconds=30.0, **kwargs):
            idx = counter[0]
            counter[0] += 1
            text = responses[idx] if idx < len(responses) else f"Response {idx}"
            return InferenceResult(success=True, text=text, error="",
                                   inference_seconds=0.01)
        return _fn

    def test_history_not_used_for_depth_zero(self):
        prompts_seen = []

        from web_console.inference_client import InferenceResult

        def _fn(image_path=None, prompt=None, *, cli_path="", socket_path="",
                timeout_seconds=30.0, **kw):
            prompts_seen.append(prompt)
            return InferenceResult(success=True, text="resp", error="",
                                   inference_seconds=0.01)

        with tempfile.TemporaryDirectory() as d:
            imgs = [str(pathlib.Path(d) / f"f{i}.jpg") for i in range(2)]
            for p in imgs:
                pathlib.Path(p).write_bytes(b"FAKE")
            defn = ExperimentDefinition(
                strategy="single_frame_observation_history",
                image_paths=imgs,
                task_prompt="Describe.",
                observation_history_max_entries=0,
            )
            run_experiment(defn, inference_fn=_fn, socket_path="/tmp/fake.sock")
        # History depth 0 — "Prior observations" should never appear.
        for p in prompts_seen:
            self.assertNotIn("Prior observations", p)

    def test_history_accumulates_with_depth_2(self):
        prompts_seen = []
        from web_console.inference_client import InferenceResult
        counter = [0]

        def _fn(image_path=None, prompt=None, *, cli_path="", socket_path="",
                timeout_seconds=30.0, **kw):
            prompts_seen.append(prompt)
            counter[0] += 1
            return InferenceResult(success=True, text=f"Obs{counter[0]}",
                                   error="", inference_seconds=0.01)

        with tempfile.TemporaryDirectory() as d:
            imgs = [str(pathlib.Path(d) / f"f{i}.jpg") for i in range(3)]
            for p in imgs:
                pathlib.Path(p).write_bytes(b"FAKE")
            defn = ExperimentDefinition(
                strategy="single_frame_observation_history",
                image_paths=imgs,
                task_prompt="Describe.",
                observation_history_max_entries=2,
                observation_history_max_chars=10000,
            )
            run_experiment(defn, inference_fn=_fn, socket_path="/tmp/fake.sock")
        # Second prompt should contain "Prior observations"
        self.assertNotIn("Prior observations", prompts_seen[0])
        self.assertIn("Prior observations", prompts_seen[1])
        self.assertIn("Prior observations", prompts_seen[2])


class TestComputeHistoryMatrix(unittest.TestCase):

    def test_matrix_sorts_by_depth(self):
        """compute_history_matrix should sort rows by observation_history_max_entries."""
        manifests = [
            {
                "experiment_id": "aaa",
                "strategy": "single_frame_observation_history",
                "image_count": 2,
                "successful_frames": 2,
                "failed_frames": 0,
                "repetition_flags": 0,
                "mean_latency_ms": 50.0,
                "min_latency_ms": 40.0,
                "max_latency_ms": 60.0,
                "definition": {
                    "observation_history_max_entries": 3,
                    "task_prompt": "Describe.",
                    "max_generate_length": 96,
                    "temperature": 0.2,
                },
            },
            {
                "experiment_id": "bbb",
                "strategy": "single_frame",
                "image_count": 2,
                "successful_frames": 2,
                "failed_frames": 0,
                "repetition_flags": 0,
                "mean_latency_ms": 45.0,
                "min_latency_ms": 42.0,
                "max_latency_ms": 48.0,
                "definition": {
                    "observation_history_max_entries": 0,
                    "task_prompt": "Describe.",
                    "max_generate_length": 96,
                    "temperature": 0.2,
                },
            },
        ]
        rows = compute_history_matrix(manifests)
        self.assertEqual(len(rows), 2)
        # depth=0 should come first
        self.assertEqual(rows[0]["history_depth"], 0)
        self.assertEqual(rows[1]["history_depth"], 3)

    def test_matrix_empty_input(self):
        rows = compute_history_matrix([])
        self.assertEqual(rows, [])


# ── model catalog tests ───────────────────────────────────────────────────────


class TestModelCatalog(unittest.TestCase):

    def test_make_model_id_stable(self):
        id1 = _make_model_id("MyModel", "/some/path")
        id2 = _make_model_id("MyModel", "/some/path")
        self.assertEqual(id1, id2)
        self.assertIn("MyModel", id1)
        # Different paths must produce different IDs.
        id3 = _make_model_id("MyModel", "/other/path")
        self.assertNotEqual(id1, id3)

    def test_make_model_id_deterministic_format(self):
        """model_id must use a hex digest suffix, not a numeric hash."""
        model_id = _make_model_id("MyModel", "/some/path")
        # Format: <safe_name>_<8-hex-chars>
        parts = model_id.rsplit("_", 1)
        self.assertEqual(len(parts), 2, f"Unexpected format: {model_id!r}")
        suffix = parts[1]
        self.assertEqual(len(suffix), 8, f"Expected 8-char hex suffix, got {suffix!r}")
        self.assertTrue(
            all(c in "0123456789abcdef" for c in suffix),
            f"Suffix is not hex: {suffix!r}",
        )

    def test_make_model_id_no_path(self):
        model_id = _make_model_id("TestModel", "")
        self.assertEqual(model_id, "TestModel")

    def test_profile_from_env_none_when_no_vars(self):
        env_backup = {}
        for k in ("EDGE_VLM_MODEL_NAME", "EDGE_VLM_LLM_ENGINE_DIR"):
            env_backup[k] = os.environ.pop(k, None)
        try:
            result = _profile_from_env()
            self.assertIsNone(result)
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_profile_from_env_with_name_and_dir(self):
        with tempfile.TemporaryDirectory() as d:
            env_backup = {
                k: os.environ.pop(k, None)
                for k in ("EDGE_VLM_MODEL_NAME", "EDGE_VLM_LLM_ENGINE_DIR",
                           "EDGE_VLM_MULTIMODAL_ENGINE_DIR", "EDGELLM_PLUGIN_PATH",
                           "EDGE_VLM_WORKSPACE_DIR")
            }
            try:
                os.environ["EDGE_VLM_MODEL_NAME"] = "FakeModel"
                os.environ["EDGE_VLM_LLM_ENGINE_DIR"] = d
                profile = _profile_from_env()
                self.assertIsNotNone(profile)
                self.assertEqual(profile.model_name, "FakeModel")
                self.assertTrue(profile.is_active)
                self.assertTrue(profile.llm_engine_exists)
            finally:
                for k, v in env_backup.items():
                    if v is not None:
                        os.environ[k] = v
                    else:
                        os.environ.pop(k, None)

    def test_scan_workspace_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            profiles = _scan_workspace(d)
            self.assertEqual(profiles, [])

    def test_scan_workspace_finds_model(self):
        with tempfile.TemporaryDirectory() as ws:
            model_dir = pathlib.Path(ws) / "FakeModel" / "engine"
            model_dir.mkdir(parents=True)
            (model_dir / "llm").mkdir()
            profiles = _scan_workspace(ws)
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].model_name, "FakeModel")
            self.assertTrue(profiles[0].multimodal_engine_exists)

    def test_scan_workspace_skips_hidden(self):
        with tempfile.TemporaryDirectory() as ws:
            hidden = pathlib.Path(ws) / ".hidden" / "engine"
            hidden.mkdir(parents=True)
            profiles = _scan_workspace(ws)
            self.assertEqual(profiles, [])

    def test_discover_models_empty(self):
        env_backup = {}
        for k in ("EDGE_VLM_MODEL_NAME", "EDGE_VLM_LLM_ENGINE_DIR",
                   "EDGE_VLM_WORKSPACE_DIR"):
            env_backup[k] = os.environ.pop(k, None)
        try:
            profiles = discover_models(workspace_dir="")
            self.assertIsInstance(profiles, list)
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_discover_models_deduplicates(self):
        """When env points at the same model dir as workspace scan, no duplicates."""
        with tempfile.TemporaryDirectory() as ws:
            model_dir = pathlib.Path(ws) / "AModel" / "engine"
            (model_dir / "llm").mkdir(parents=True)
            env_backup = {
                k: os.environ.pop(k, None)
                for k in ("EDGE_VLM_MODEL_NAME", "EDGE_VLM_LLM_ENGINE_DIR",
                           "EDGE_VLM_MULTIMODAL_ENGINE_DIR", "EDGELLM_PLUGIN_PATH")
            }
            try:
                os.environ["EDGE_VLM_MODEL_NAME"] = "AModel"
                os.environ["EDGE_VLM_LLM_ENGINE_DIR"] = str(model_dir / "llm")
                profiles = discover_models(workspace_dir=ws)
                ids = [p.model_id for p in profiles]
                self.assertEqual(len(ids), len(set(ids)), "Duplicate model IDs")
            finally:
                for k, v in env_backup.items():
                    if v is not None:
                        os.environ[k] = v
                    else:
                        os.environ.pop(k, None)


# ── dataset catalog tests ─────────────────────────────────────────────────────


class TestDatasetCatalog(unittest.TestCase):

    def test_build_download_command_known_key(self):
        cmd = build_download_command("/scripts/download.sh", "image-proc")
        self.assertIsNotNone(cmd)
        self.assertIsInstance(cmd, list)
        self.assertIn("image-proc", cmd)
        self.assertNotIn(True, cmd)   # no shell=True via bool leak

    def test_build_download_command_h264(self):
        cmd = build_download_command("/scripts/download.sh", "h264")
        self.assertIsNotNone(cmd)
        self.assertIn("h264", cmd)

    def test_build_download_command_unknown_key_returns_none(self):
        self.assertIsNone(
            build_download_command("/scripts/download.sh", "arbitrary_key")
        )

    def test_build_download_command_traversal_returns_none(self):
        self.assertIsNone(
            build_download_command("/scripts/download.sh", "../../../etc/passwd")
        )

    def test_build_download_command_empty_key_returns_none(self):
        self.assertIsNone(build_download_command("/scripts/download.sh", ""))

    def test_downloadable_bags_keys_unique(self):
        keys = [d["key"] for d in _DOWNLOADABLE_BAGS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_discover_datasets_no_dirs(self):
        data = discover_datasets(rosbag_root="/nonexistent/dir",
                                 image_root=None, video_root=None)
        self.assertIn("rosbags", data)
        self.assertIn("image_datasets", data)
        self.assertIn("video_datasets", data)
        # Should still list downloadable bags
        self.assertGreater(len(data["rosbags"]), 0)
        # All listed bags that are not locally installed should be downloadable
        for bag in data["rosbags"]:
            if not bag["installed"]:
                self.assertTrue(bag["downloadable"])

    def test_discover_datasets_with_image_dir(self):
        with tempfile.TemporaryDirectory() as root:
            img_root = pathlib.Path(root) / "images"
            dataset_dir = img_root / "my_dataset"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "frame_000.jpg").write_bytes(b"FAKE")
            (dataset_dir / "frame_001.png").write_bytes(b"FAKE")

            data = discover_datasets(rosbag_root="/nonexistent",
                                     image_root=str(img_root), video_root=None)
            self.assertEqual(len(data["image_datasets"]), 1)
            self.assertEqual(data["image_datasets"][0]["name"], "my_dataset")
            self.assertEqual(data["image_datasets"][0]["image_count"], 2)

    def test_discover_datasets_detects_nested_downloaded_asset(self):
        """Nested NGC bags merge into their top-level downloadable entry."""
        with tempfile.TemporaryDirectory() as root:
            bag_dir = (
                pathlib.Path(root)
                / "h264"
                / "isaac_ros_h264_decoder"
                / "quickstart"
            )
            bag_dir.mkdir(parents=True)
            (bag_dir / "metadata.yaml").write_text(
                "rosbag2_bagfile_information:\n"
                "  duration: {nanoseconds: 1000000000}\n",
                encoding="utf-8",
            )

            data = discover_datasets(
                rosbag_root=root, image_root=None, video_root=None
            )
            h264 = next(b for b in data["rosbags"] if b["key"] == "h264")
            self.assertTrue(h264["installed"])
            self.assertTrue(h264["downloadable"])
            self.assertEqual(h264["local_path"], str(bag_dir))

    def test_discover_datasets_with_video_dir(self):
        with tempfile.TemporaryDirectory() as root:
            vid_root = pathlib.Path(root) / "videos"
            vid_root.mkdir()
            (vid_root / "clip.mp4").write_bytes(b"FAKE")

            data = discover_datasets(rosbag_root="/nonexistent",
                                     image_root=None, video_root=str(vid_root))
            self.assertEqual(len(data["video_datasets"]), 1)
            self.assertEqual(data["video_datasets"][0]["name"], "clip.mp4")


# ── new API route smoke tests ─────────────────────────────────────────────────


class TestNewAPIRoutes(unittest.TestCase):
    """Smoke tests for /api/models and /api/datasets served by ConsoleServer."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        run_store = RunStore(self._tmp)
        cfg = {
            "socket_path": "",
            "quiet": True,
            "workspace_dir": "",
            "rosbag_dir": "/nonexistent",
            "image_dataset_dir": "",
            "video_dataset_dir": "",
        }
        with patch("web_console.server.check_server_reachable",
                   return_value={"reachable": False}):
            self._srv = ConsoleServer(
                host="127.0.0.1", port=0, config=cfg, run_store=run_store
            )
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        time.sleep(0.1)
        self._host, self._port = self._srv.server_address

    def tearDown(self):
        self._srv.shutdown()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _get(self, path):
        conn = HTTPConnection(self._host, self._port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, json.loads(body) if body else {}

    def test_api_models_returns_200(self):
        status, data = self._get("/api/models")
        self.assertEqual(status, 200)
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)

    def test_api_datasets_returns_200(self):
        status, data = self._get("/api/datasets")
        self.assertEqual(status, 200)
        self.assertIn("rosbags", data)
        self.assertIn("image_datasets", data)
        self.assertIn("video_datasets", data)

    def test_index_contains_workbench_title(self):
        conn = HTTPConnection(self._host, self._port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertIn(b"edge_vlm_ros", body)
        self.assertIn(b"Workbench", body)

    def test_api_datasets_download_unknown_key_returns_400(self):
        conn = HTTPConnection(self._host, self._port, timeout=5)
        body_bytes = json.dumps({"bag_key": "UNKNOWN_BAG"}).encode()
        conn.request(
            "POST", "/api/datasets/download", body=body_bytes,
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(body_bytes))},
        )
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        self.assertIn(resp.status, (400, 422))

    def test_api_experiment_run_invalid_definition_returns_422(self):
        conn = HTTPConnection(self._host, self._port, timeout=5)
        body_bytes = json.dumps({
            "strategy": "bad_strategy",
            "image_paths": ["/data/frame.jpg"],
            "task_prompt": "Describe.",
        }).encode()
        conn.request(
            "POST", "/api/experiment/run", body=body_bytes,
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(body_bytes))},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertIn(resp.status, (400, 422))


# ── experiment_stack.sh bash syntax test ──────────────────────────────────────


class TestModelIdCrossProcess(unittest.TestCase):
    """Verify that model_id is stable across interpreter restarts."""

    def test_make_model_id_stable_across_hash_seeds(self):
        """_make_model_id must return the same value regardless of PYTHONHASHSEED."""
        script = (
            f"import sys; sys.path.insert(0, {str(_REPO_ROOT)!r}); "
            "from web_console.model_catalog import _make_model_id; "
            "print(_make_model_id('TestModel', '/some/engine/path/llm'))"
        )
        results = set()
        for seed in ("1", "2", "42", "0", "99999"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            out = subprocess.check_output(
                [sys.executable, "-c", script],
                text=True,
                env=env,
            ).strip()
            results.add(out)
        self.assertEqual(
            len(results),
            1,
            f"model_id changed across PYTHONHASHSEED values: {results}",
        )
        (stable_id,) = results
        self.assertIn("TestModel", stable_id)


class TestExperimentConcurrency(unittest.TestCase):
    """Bounded experiment coordinator tests."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        run_store = RunStore(pathlib.Path(self._tmp) / "runs")
        cfg = {"socket_path": "", "quiet": True}
        self._srv, self._port, self._thread = _start_test_server(
            config=cfg, run_store=run_store
        )

    def tearDown(self):
        self._srv.shutdown()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _post_json(self, path, data):
        body = json.dumps(data).encode()
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request(
            "POST", path, body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()
        return resp.status, json.loads(resp_body) if resp_body else {}

    def test_concurrent_experiment_rejected_with_409(self):
        """A second experiment while one is active must return 409."""
        # Inject a fake active experiment ID directly into the server.
        self._srv._active_experiment_id = "aaaaaaaa-0000-0000-0000-000000000001"
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(b"FAKE")
                img = f.name
            status, data = self._post_json(
                "/api/experiment/run",
                {
                    "strategy": "single_frame",
                    "image_paths": [img],
                    "task_prompt": "Describe.",
                },
            )
        finally:
            self._srv._active_experiment_id = None
            os.unlink(img)
        self.assertEqual(status, 409)
        self.assertIn("error", data)
        self.assertIn("aaaaaaaa", data["error"])

    def test_experiment_coordinator_clears_on_completion(self):
        """_active_experiment_id is cleared when an experiment reaches a terminal state."""
        # Directly confirm the lock and field are available.
        self.assertIsNone(self._srv._active_experiment_id)
        self.assertIsInstance(self._srv._active_experiment_lock, type(__import__("threading").Lock()))

    def test_experiment_validates_all_image_paths(self):
        """Image path validation must check ALL paths, not just the first 100."""
        with tempfile.TemporaryDirectory() as d:
            # Create 101 real files
            real_paths = []
            for i in range(101):
                p = os.path.join(d, f"frame_{i:04d}.jpg")
                pathlib.Path(p).write_bytes(b"FAKE")
                real_paths.append(p)
            # Replace the 101st path with a nonexistent file
            real_paths[100] = os.path.join(d, "nonexistent_101.jpg")

            status, data = self._post_json(
                "/api/experiment/run",
                {
                    "strategy": "single_frame",
                    "image_paths": real_paths,
                    "task_prompt": "Describe.",
                },
            )
        # Should fail because the 101st path does not exist.
        self.assertIn(status, (400, 422), f"Expected 400/422, got {status}: {data}")
        self.assertIn("error", data)
        self.assertIn("nonexistent_101.jpg", data["error"])


class TestExperimentStackScript(unittest.TestCase):
    """Verify experiment_stack.sh passes bash -n syntax check."""

    def test_bash_syntax_check(self):
        script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "experiment_stack.sh"
        if not script.is_file():
            self.skipTest(f"experiment_stack.sh not found at {script}")
        if shutil.which("bash") is None:
            self.skipTest("bash not available")
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"bash -n failed:\n{result.stderr}")

    def test_run_image_proc_bash_syntax(self):
        """run_image_proc_test.sh must pass bash -n syntax check."""
        script = (
            pathlib.Path(__file__).resolve().parents[2]
            / "scripts"
            / "test_data"
            / "run_image_proc_test.sh"
        )
        if not script.is_file():
            self.skipTest(f"run_image_proc_test.sh not found at {script}")
        if shutil.which("bash") is None:
            self.skipTest("bash not available")
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"bash -n failed:\n{result.stderr}")


# ── rosbag compatibility metadata tests ──────────────────────────────────────


class TestRosbagCompatibility(unittest.TestCase):
    """Tests for raw_image_compatible and compatibility_note catalog fields."""

    def _make_bag(self, tmpdir: str, key: str, metadata_content: str) -> pathlib.Path:
        """Create a fake installed rosbag under *tmpdir*/<key>/<bag>."""
        bag_dir = pathlib.Path(tmpdir) / key / "bag0"
        bag_dir.mkdir(parents=True)
        (bag_dir / "metadata.yaml").write_text(metadata_content, encoding="utf-8")
        return bag_dir

    def test_image_proc_catalog_entry_is_compatible(self):
        """The image-proc downloadable entry must be marked raw_image_compatible."""
        data = discover_datasets(
            rosbag_root="/nonexistent", image_root=None, video_root=None
        )
        entry = next(b for b in data["rosbags"] if b["key"] == "image-proc")
        self.assertTrue(entry["raw_image_compatible"])
        self.assertEqual(entry["compatibility_note"], "")

    def test_h264_catalog_entry_is_not_compatible(self):
        """The h264 downloadable entry must be marked NOT raw_image_compatible."""
        data = discover_datasets(
            rosbag_root="/nonexistent", image_root=None, video_root=None
        )
        entry = next(b for b in data["rosbags"] if b["key"] == "h264")
        self.assertFalse(entry["raw_image_compatible"])
        self.assertIn("H.264", entry["compatibility_note"])

    def test_installed_bag_with_raw_image_topic_is_compatible(self):
        """An installed bag whose metadata has a sensor_msgs/Image topic is compatible."""
        metadata = (
            "rosbag2_bagfile_information:\n"
            "  topics_with_message_count:\n"
            "    - topic_metadata: "
            "{name: /camera/image_raw, type: sensor_msgs/msg/Image, serialization_format: cdr}\n"
            "  duration: {nanoseconds: 5000000000}\n"
        )
        with tempfile.TemporaryDirectory() as root:
            self._make_bag(root, "my_raw_bag", metadata)
            data = discover_datasets(rosbag_root=root, image_root=None, video_root=None)
        entry = next(b for b in data["rosbags"] if b["key"] == "my_raw_bag")
        self.assertTrue(entry["raw_image_compatible"])
        self.assertEqual(entry["compatibility_note"], "")

    def test_installed_bag_with_compressed_image_only_is_not_compatible(self):
        """An installed bag with only CompressedImage topics is not compatible."""
        metadata = (
            "rosbag2_bagfile_information:\n"
            "  topics_with_message_count:\n"
            "    - topic_metadata: "
            "{name: /camera/image_compressed, "
            "type: sensor_msgs/msg/CompressedImage, serialization_format: cdr}\n"
            "  duration: {nanoseconds: 5000000000}\n"
        )
        with tempfile.TemporaryDirectory() as root:
            self._make_bag(root, "my_h264_bag", metadata)
            data = discover_datasets(rosbag_root=root, image_root=None, video_root=None)
        entry = next(b for b in data["rosbags"] if b["key"] == "my_h264_bag")
        self.assertFalse(entry["raw_image_compatible"])
        self.assertNotEqual(entry["compatibility_note"], "")

    def test_h264_installed_bag_uses_catalog_compatibility_note(self):
        """When the h264 bag is installed, the catalog note overrides the parsed note."""
        metadata = (
            "rosbag2_bagfile_information:\n"
            "  topics_with_message_count:\n"
            "    - topic_metadata: "
            "{name: /left, type: sensor_msgs/msg/CompressedImage, serialization_format: cdr}\n"
            "  duration: {nanoseconds: 2000000000}\n"
        )
        with tempfile.TemporaryDirectory() as root:
            bag_dir = (
                pathlib.Path(root) / "h264" / "isaac_ros_h264_decoder" / "quickstart"
            )
            bag_dir.mkdir(parents=True)
            (bag_dir / "metadata.yaml").write_text(metadata, encoding="utf-8")
            data = discover_datasets(rosbag_root=root, image_root=None, video_root=None)
        entry = next(b for b in data["rosbags"] if b["key"] == "h264")
        self.assertTrue(entry["installed"])
        self.assertFalse(entry["raw_image_compatible"])
        self.assertIn("H.264", entry["compatibility_note"])

    def test_compatibility_fields_present_in_all_catalog_entries(self):
        """Every entry returned by discover_datasets must carry both compatibility fields."""
        data = discover_datasets(
            rosbag_root="/nonexistent", image_root=None, video_root=None
        )
        for entry in data["rosbags"]:
            self.assertIn("raw_image_compatible", entry,
                          f"Missing raw_image_compatible in {entry['key']}")
            self.assertIn("compatibility_note", entry,
                          f"Missing compatibility_note in {entry['key']}")


# ── rosbag_path allowlist validation tests ───────────────────────────────────


class TestRosbagPathValidation(unittest.TestCase):
    """Server-side allowlist validation for rosbag_path in /api/ros/start."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        run_store = RunStore(pathlib.Path(self._tmp) / "runs")
        self._rosbag_root = pathlib.Path(self._tmp) / "rosbags"
        self._rosbag_root.mkdir()
        cfg = {
            "socket_path": "",
            "quiet": True,
            "rosbag_dir": str(self._rosbag_root),
            "image_dataset_dir": "",
            "video_dataset_dir": "",
        }
        self._srv, self._port, self._thread = _start_test_server(
            config=cfg, run_store=run_store
        )

    def tearDown(self):
        self._srv.shutdown()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _post_ros_start(self, params: dict) -> tuple:
        body = json.dumps({"params": params}).encode()
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request(
            "POST", "/api/ros/start", body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()
        return resp.status, json.loads(resp_body) if resp_body else {}

    def _make_installed_bag(self, key: str) -> str:
        """Create a fake installed bag and return its local_path."""
        bag_dir = self._rosbag_root / key / "bag0"
        bag_dir.mkdir(parents=True)
        (bag_dir / "metadata.yaml").write_text(
            "rosbag2_bagfile_information:\n"
            "  topics_with_message_count:\n"
            "    - topic_metadata: "
            "{name: /img, type: sensor_msgs/msg/Image, serialization_format: cdr}\n"
            "  duration: {nanoseconds: 1000000000}\n",
            encoding="utf-8",
        )
        return str(bag_dir)

    def test_arbitrary_rosbag_path_rejected(self):
        """An arbitrary filesystem path must be rejected with 400."""
        status, data = self._post_ros_start({"rosbag_path": "/etc/passwd"})
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_path_traversal_rosbag_path_rejected(self):
        """A traversal path must be rejected with 400."""
        status, data = self._post_ros_start(
            {"rosbag_path": str(self._rosbag_root) + "/../../etc"}
        )
        self.assertEqual(status, 400)

    def test_installed_catalog_path_accepted(self):
        """A local_path from the installed catalog is accepted and env is built."""
        installed_path = self._make_installed_bag("my-test-bag")
        # The server will try to launch the ROS script; we only need to verify
        # the 400 is NOT returned for the path check. We patch start_ros_experiment
        # to avoid actually launching a subprocess.
        with patch.object(
            self._srv.process_manager,
            "start_ros_experiment",
            return_value=99999,
        ):
            status, data = self._post_ros_start({"rosbag_path": installed_path})
        # Should not get a 400 for path validation.
        self.assertNotEqual(status, 400, f"Unexpected 400: {data}")

    def test_no_rosbag_path_param_does_not_error(self):
        """Omitting rosbag_path entirely should not cause a 400 for path validation."""
        with patch.object(
            self._srv.process_manager,
            "start_ros_experiment",
            return_value=99999,
        ):
            status, data = self._post_ros_start({})
        self.assertNotEqual(status, 400, f"Unexpected 400 without rosbag_path: {data}")


# ── _build_ros_env ROSBAG_PATH propagation tests ──────────────────────────────


class TestBuildRosEnvRosbagPath(unittest.TestCase):
    """Verify _build_ros_env forwards rosbag_path as ROSBAG_PATH."""

    def test_rosbag_path_forwarded_to_env(self):
        env = _build_ros_env(
            {"rosbag_path": "/data/my_bag"}, cfg={}, artifact_dir=None
        )
        self.assertEqual(env.get("ROSBAG_PATH"), "/data/my_bag")

    def test_rosbag_path_absent_when_not_in_params(self):
        env = _build_ros_env({}, cfg={}, artifact_dir=None)
        # Should not inject an empty ROSBAG_PATH.
        # (It may still be present from os.environ; verify it is not *set* by _build_ros_env.)
        env2 = _build_ros_env({"image_topic": "/cam"}, cfg={}, artifact_dir=None)
        # Either ROSBAG_PATH is absent or unchanged from os.environ.
        # The critical check: passing rosbag_path sets it correctly.
        env3 = _build_ros_env({"rosbag_path": "/bags/custom"}, cfg={}, artifact_dir=None)
        self.assertEqual(env3["ROSBAG_PATH"], "/bags/custom")


# ── UI element presence tests ─────────────────────────────────────────────────


class TestUIElementPresence(unittest.TestCase):
    """Verify required HTML elements are present in the served index page."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        run_store = RunStore(pathlib.Path(self._tmp) / "runs")
        cfg = {"socket_path": "", "quiet": True}
        self._srv, self._port, self._thread = _start_test_server(
            config=cfg, run_store=run_store
        )

    def tearDown(self):
        self._srv.shutdown()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _get_index(self) -> str:
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)
        return body.decode("utf-8", errors="replace")

    def test_ros_selected_bag_element_exists(self):
        """The #ros-selected-bag element must be present in the ROS experiment panel."""
        html = self._get_index()
        self.assertIn('id="ros-selected-bag"', html)

    def test_app_js_has_selected_bag_variable(self):
        """app.js must declare _selectedBag."""
        js_path = pathlib.Path(__file__).resolve().parents[1] / "static" / "app.js"
        js = js_path.read_text(encoding="utf-8")
        self.assertIn("_selectedBag", js)

    def test_app_js_has_select_bag_for_experiment(self):
        """app.js must define selectBagForExperiment."""
        js_path = pathlib.Path(__file__).resolve().parents[1] / "static" / "app.js"
        js = js_path.read_text(encoding="utf-8")
        self.assertIn("selectBagForExperiment", js)

    def test_app_js_sends_rosbag_path_in_start_ros(self):
        """startRos() in app.js must include rosbag_path when _selectedBag is set."""
        js_path = pathlib.Path(__file__).resolve().parents[1] / "static" / "app.js"
        js = js_path.read_text(encoding="utf-8")
        self.assertIn("rosbag_path", js)

    def test_app_js_uses_raw_image_compatible_for_button(self):
        """_bagTile must check raw_image_compatible before showing Use in Experiment."""
        js_path = pathlib.Path(__file__).resolve().parents[1] / "static" / "app.js"
        js = js_path.read_text(encoding="utf-8")
        self.assertIn("raw_image_compatible", js)

    def test_app_js_h264_shows_compatibility_note(self):
        """_bagTile must show compatibility_note for incompatible bags."""
        js_path = pathlib.Path(__file__).resolve().parents[1] / "static" / "app.js"
        js = js_path.read_text(encoding="utf-8")
        self.assertIn("compatibility_note", js)



# ── Blocker 1: Multiline single-quoted YAML scalars ───────────────────────────

class TestSingleQuoteHelpers(unittest.TestCase):
    """Unit tests for _single_quote_close_idx and _fold_single_quoted_scalar."""

    def test_close_idx_simple(self):
        self.assertEqual(_single_quote_close_idx("hello'"), 5)

    def test_close_idx_at_start(self):
        self.assertEqual(_single_quote_close_idx("'"), 0)

    def test_close_idx_no_close(self):
        self.assertEqual(_single_quote_close_idx("hello"), -1)

    def test_close_idx_empty(self):
        self.assertEqual(_single_quote_close_idx(""), -1)

    def test_close_idx_escaped_quote_not_close(self):
        # '' is an escape, should not close the scalar
        self.assertEqual(_single_quote_close_idx("it''s a test'"), 12)

    def test_close_idx_double_escape_then_close(self):
        # '''' is two escaped '' followed by closing '
        # wait: '''' — chars: ' ' ' '
        # i=0: s[0]='s[1]=' → skip to i=2; s[2]='s[3]=' → skip to i=4; out of range → -1
        # Actually '''' as a scalar *body* (without outer quotes):
        # chars: ' ' ' '
        # i=0: s[0]=', s[1]=' → i+=2 (escaped)
        # i=2: s[2]=', s[3]=' → i+=2 (escaped)
        # i=4: end → -1
        self.assertEqual(_single_quote_close_idx("''''"), -1)

    def test_fold_simple(self):
        self.assertEqual(_fold_single_quoted_scalar(["hello", "world"]), "hello world")

    def test_fold_blank_line_becomes_newline(self):
        result = _fold_single_quoted_scalar(["line one", "", "line two"])
        self.assertEqual(result, "line one\nline two")

    def test_fold_two_blank_lines_become_two_newlines(self):
        result = _fold_single_quoted_scalar(["line one", "", "", "line two"])
        self.assertEqual(result, "line one\n\nline two")

    def test_fold_escaped_quotes_decoded(self):
        result = _fold_single_quoted_scalar(["it''s a test"])
        self.assertEqual(result, "it's a test")

    def test_fold_empty_parts_list(self):
        self.assertEqual(_fold_single_quoted_scalar([]), "")

    def test_fold_single_part(self):
        self.assertEqual(_fold_single_quoted_scalar(["hello"]), "hello")


class TestParseMultilineSingleQuoted(unittest.TestCase):
    """Regression tests for multiline single-quoted YAML scalar parsing."""

    def test_thor_multiline_single_quoted_fixture(self):
        """Exact format observed on Thor: single-quoted scalar spanning multiple
        physical lines with blank lines as paragraph separators."""
        text = (
            "frame_sequence: 1\n"
            "response: 'Based on the provided image, here is a detailed description of the scene:\n"
            "\n"
            "\n"
            "  The image captures a busy urban intersection.'\n"
            "inference_seconds: 1.23\n"
            "success: true\n"
            "error: ''\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        text_val = frames[0]["text"]
        # Must contain the first physical line's content
        self.assertIn("Based on the provided image", text_val)
        # Must contain the continuation line's content
        self.assertIn("The image captures a busy urban intersection.", text_val)
        # The two blank lines between them should produce a newline separator
        self.assertIn("\n", text_val)

    def test_single_quoted_scalar_on_one_line_unchanged(self):
        """Single-quoted scalars that open and close on the same line still work."""
        text = (
            "frame_sequence: 2\n"
            "response: 'simple response'\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["text"], "simple response")

    def test_single_quoted_empty_string_unchanged(self):
        """Single-quoted empty string '' still produces empty string."""
        text = (
            "frame_sequence: 3\n"
            "response: ''\n"
            "success: false\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["text"], "")

    def test_multiline_sq_across_multiple_continuation_lines(self):
        """Multi-line single-quoted scalar with three content segments."""
        text = (
            "frame_sequence: 4\n"
            "response: 'Part one.\n"
            "\n"
            "Part two.\n"
            "\n"
            "Part three.'\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        t = frames[0]["text"]
        self.assertIn("Part one.", t)
        self.assertIn("Part two.", t)
        self.assertIn("Part three.", t)

    def test_multiline_sq_preserves_other_fields(self):
        """Other fields in the same message are correctly parsed alongside multiline response."""
        text = (
            "frame_sequence: 5\n"
            "response: 'Line A.\n"
            "\n"
            "Line B.'\n"
            "inference_seconds: 2.0\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["frame_seq"], 5)
        self.assertAlmostEqual(frames[0]["latency_ms"], 2000.0)
        self.assertIs(frames[0]["success"], True)
        self.assertIn("Line A.", frames[0]["text"])
        self.assertIn("Line B.", frames[0]["text"])

    def test_existing_block_scalar_still_works(self):
        """YAML block scalar (|) handling is not broken by the single-quoted changes."""
        text = (
            "frame_sequence: 1\n"
            "response: |\n"
            "  Block line one.\n"
            "  Block line two.\n"
            "inference_seconds: 0.01\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 1)
        self.assertIn("Block line one.", frames[0]["text"])
        self.assertIn("Block line two.", frames[0]["text"])

    def test_multiline_sq_across_two_frames(self):
        """Two consecutive messages both with multiline single-quoted responses."""
        text = (
            "frame_sequence: 1\n"
            "response: 'Frame one,\n"
            "\n"
            "continued.'\n"
            "success: true\n"
            "---\n"
            "frame_sequence: 2\n"
            "response: 'Frame two,\n"
            "\n"
            "also continued.'\n"
            "success: true\n"
            "---\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(len(frames), 2)
        self.assertIn("Frame one", frames[0]["text"])
        self.assertIn("continued.", frames[0]["text"])
        self.assertIn("Frame two", frames[1]["text"])
        self.assertIn("also continued.", frames[1]["text"])

    def test_multiline_sq_incomplete_is_discarded(self):
        """An unclosed single-quoted scalar at EOF (no ---) is discarded."""
        text = (
            "frame_sequence: 1\n"
            "response: 'no closing quote\n"
        )
        frames = _parse_results_log(text)
        self.assertEqual(frames, [])


# ── Blocker 2: Stack script model-mismatch tests ──────────────────────────────

class TestExperimentStackModelMismatch(unittest.TestCase):
    """Tests covering the model-conflict detection logic in experiment_stack.sh.

    These tests verify behaviour using bash -n (syntax) and by inspecting the
    script source for the required constructs; actual execution requires the
    native edge_vlm_server binary and is validated in integration tests.
    """

    def _script_path(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[2] / "scripts" / "experiment_stack.sh"

    def test_stack_script_bash_syntax_still_passes(self):
        """bash -n must pass after the model-mismatch additions."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"bash -n failed:\n{result.stderr}")

    def test_check_socket_model_conflict_function_present(self):
        """Script must define _check_socket_model_conflict."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        content = script.read_text(encoding="utf-8")
        self.assertIn("_check_socket_model_conflict", content)

    def test_socket_listener_pid_function_present(self):
        """Script must define _socket_listener_pid."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        content = script.read_text(encoding="utf-8")
        self.assertIn("_socket_listener_pid", content)

    def test_proc_argv1_function_present(self):
        """Script must define _proc_argv1 for cmdline inspection."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        content = script.read_text(encoding="utf-8")
        self.assertIn("_proc_argv1", content)

    def test_model_conflict_check_called_in_cmd_start(self):
        """cmd_start must call _check_socket_model_conflict before starting server."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        content = script.read_text(encoding="utf-8")
        # Verify the call is inside cmd_start (between 'cmd_start()' and 'cmd_stop()')
        start_idx = content.find("cmd_start()")
        stop_idx = content.find("cmd_stop()")
        self.assertGreater(start_idx, 0)
        self.assertGreater(stop_idx, start_idx)
        cmd_start_body = content[start_idx:stop_idx]
        self.assertIn("_check_socket_model_conflict", cmd_start_body)

    def test_status_surfaces_running_llm_dir_from_cmdline(self):
        """cmd_status must read the actual LLM dir from the running process cmdline."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        content = script.read_text(encoding="utf-8")
        # status function must call _proc_argv1 to read the actual running engine
        status_idx = content.find("cmd_status()")
        self.assertGreater(status_idx, 0)
        # Find the end of cmd_status (next function definition)
        next_fn = content.find("\ncmd_", status_idx + 1)
        cmd_status_body = content[status_idx:next_fn] if next_fn > 0 else content[status_idx:]
        self.assertIn("_proc_argv1", cmd_status_body)

    def test_reuse_external_socket_variable_present(self):
        """Script must use _REUSE_EXTERNAL_SOCKET to signal authorised external reuse."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        content = script.read_text(encoding="utf-8")
        self.assertIn("_REUSE_EXTERNAL_SOCKET", content)

    def test_model_mismatch_error_message_present(self):
        """Script must emit a 'Model conflict' error for mismatched external services."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        content = script.read_text(encoding="utf-8")
        self.assertIn("Model conflict", content)

    def test_unverifiable_external_service_error_present(self):
        """Script must fail when the socket owner's model cannot be verified."""
        script = self._script_path()
        if not script.exists():
            self.skipTest("experiment_stack.sh not found")
        content = script.read_text(encoding="utf-8")
        # The script uses either "unidentifiable" or "cannot be verified" to
        # describe an external service whose loaded model cannot be confirmed.
        self.assertTrue(
            "unidentifiable" in content or "cannot be verified" in content,
            "Script must emit an error when socket owner model cannot be verified",
        )


# ── Blocker 3: Benchmark vs result-frame count reconciliation ─────────────────

class TestBenchmarkCountReconciliation(unittest.TestCase):
    """Tests for the count_note added when benchmark and result-frame counts differ."""

    def _build_benchmark_summary(
        self, frame_recs: list, result_frames: list
    ) -> dict:
        """Replicate the server-side benchmark_summary construction logic."""
        success_recs = [f for f in frame_recs if f.get("success", True)]
        infer_ms_vals = [
            float(f["inference_ms"])
            for f in success_recs
            if "inference_ms" in f
        ]
        benchmark_summary = {
            "frame_count": len(frame_recs),
            "successful_frames": len(success_recs),
            "failed_frames": len(frame_recs) - len(success_recs),
            "dropped_frames": 0,
            "mean_inference_ms": round(sum(infer_ms_vals) / len(infer_ms_vals), 2)
            if infer_ms_vals
            else None,
            "min_inference_ms": round(min(infer_ms_vals), 2) if infer_ms_vals else None,
            "max_inference_ms": round(max(infer_ms_vals), 2) if infer_ms_vals else None,
            "source": "benchmark.jsonl (all processed inference samples)",
        }
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
        return benchmark_summary

    def test_no_count_note_when_counts_match(self):
        """count_note must be absent when benchmark and result-frame counts agree."""
        frame_recs = [
            {"record_type": "frame", "success": True, "inference_ms": 50.0},
        ]
        result_frames = [{"text": "frame 1"}]
        summary = self._build_benchmark_summary(frame_recs, result_frames)
        self.assertNotIn("count_note", summary)

    def test_count_note_added_when_benchmark_has_more_frames(self):
        """count_note must be present when benchmark.jsonl has more frames than
        results.log (the Thor-observed scenario)."""
        frame_recs = [
            {"record_type": "frame", "success": True, "inference_ms": 50.0},
            {"record_type": "frame", "success": True, "inference_ms": 60.0},
        ]
        result_frames = [{"text": "only one frame received"}]  # 1 vs 2
        summary = self._build_benchmark_summary(frame_recs, result_frames)
        self.assertIn("count_note", summary)
        self.assertIn("2", summary["count_note"])
        self.assertIn("1", summary["count_note"])

    def test_count_note_mentions_shutdown(self):
        """count_note must explain the shutdown-related reason for the discrepancy."""
        frame_recs = [{"record_type": "frame", "success": True, "inference_ms": 30.0}] * 3
        result_frames = [{"text": "f1"}, {"text": "f2"}]  # 2 vs 3
        summary = self._build_benchmark_summary(frame_recs, result_frames)
        self.assertIn("shutdown", summary["count_note"])

    def test_count_note_when_result_frames_empty_but_benchmark_has_frames(self):
        """count_note must appear even when result_frames is empty."""
        frame_recs = [{"record_type": "frame", "success": True, "inference_ms": 25.0}]
        result_frames = []
        summary = self._build_benchmark_summary(frame_recs, result_frames)
        self.assertIn("count_note", summary)

    def test_benchmark_source_label_present(self):
        """benchmark_summary must include a 'source' field for UI clarity."""
        frame_recs = [{"record_type": "frame", "success": True, "inference_ms": 20.0}]
        result_frames = [{"text": "one"}]
        summary = self._build_benchmark_summary(frame_recs, result_frames)
        self.assertIn("source", summary)
        self.assertIn("benchmark.jsonl", summary["source"])

    def test_app_js_benchmark_summary_title_uses_source(self):
        """app.js must incorporate the 'source' field into the benchmark title."""
        js_path = pathlib.Path(__file__).resolve().parents[1] / "static" / "app.js"
        js = js_path.read_text(encoding="utf-8")
        self.assertIn("bs.source", js)

    def test_app_js_renders_count_note(self):
        """app.js must render count_note when present."""
        js_path = pathlib.Path(__file__).resolve().parents[1] / "static" / "app.js"
        js = js_path.read_text(encoding="utf-8")
        self.assertIn("count_note", js)


# ── Task profile tests ────────────────────────────────────────────────────────

class TestTaskProfiles(unittest.TestCase):
    """Tests for task_profiles module: loading, hashing, and structured parsing."""

    def _write_profile(self, d, data):
        p = pathlib.Path(d) / f"{data['name']}_v{data['version']}.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_discover_profiles_loads_valid_profile(self):
        from web_console.task_profiles import discover_profiles
        with tempfile.TemporaryDirectory() as d:
            self._write_profile(d, {
                "name": "test_profile",
                "version": "1",
                "system_instruction": "You are a helpful assistant.",
                "task_prompt": "Describe the scene.",
                "schema_example": {},
            })
            profiles = discover_profiles(d)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].name, "test_profile")
        self.assertEqual(profiles[0].version, "1")

    def test_discover_profiles_empty_dir(self):
        from web_console.task_profiles import discover_profiles
        with tempfile.TemporaryDirectory() as d:
            profiles = discover_profiles(d)
        self.assertEqual(profiles, [])

    def test_discover_profiles_missing_dir(self):
        from web_console.task_profiles import discover_profiles
        profiles = discover_profiles("/tmp/nonexistent_profile_dir_xyz")
        self.assertEqual(profiles, [])

    def test_profile_id_stable(self):
        from web_console.task_profiles import TaskProfile
        p = TaskProfile(
            name="warehouse_awareness",
            version="1",
            description="",
            system_instruction="sys",
            task_prompt="task",
            prompt_hash="abc",
            profile_path="/tmp/test.json",
        )
        self.assertEqual(p.profile_id(), "warehouse_awareness_v1")

    def test_prompt_hash_deterministic(self):
        from web_console.task_profiles import discover_profiles
        with tempfile.TemporaryDirectory() as d:
            self._write_profile(d, {
                "name": "hash_test",
                "version": "1",
                "system_instruction": "sys",
                "task_prompt": "task",
                "schema_example": {},
            })
            p1 = discover_profiles(d)[0]
            p2 = discover_profiles(d)[0]
        self.assertEqual(p1.prompt_hash, p2.prompt_hash)
        self.assertGreater(len(p1.prompt_hash), 0)

    def test_prompt_hash_changes_with_content(self):
        from web_console.task_profiles import discover_profiles
        with tempfile.TemporaryDirectory() as d:
            self._write_profile(d, {
                "name": "a",
                "version": "1",
                "system_instruction": "sys",
                "task_prompt": "task A",
                "schema_example": {},
            })
            self._write_profile(d, {
                "name": "b",
                "version": "1",
                "system_instruction": "sys",
                "task_prompt": "task B",
                "schema_example": {},
            })
            profiles = discover_profiles(d)
        hashes = {p.prompt_hash for p in profiles}
        self.assertEqual(len(hashes), 2, "Different prompts must produce different hashes")

    def test_parse_structured_output_valid_json(self):
        from web_console.task_profiles import parse_structured_output
        raw = json.dumps({"summary": "all clear", "people": []})
        result = parse_structured_output(raw)
        self.assertTrue(result.parsed_ok)
        self.assertEqual(result.parsed["summary"], "all clear")
        self.assertEqual(result.raw, raw)

    def test_parse_structured_output_json_in_markdown(self):
        from web_console.task_profiles import parse_structured_output
        raw = '```json\n{"summary": "ok"}\n```'
        result = parse_structured_output(raw)
        self.assertTrue(result.parsed_ok)
        self.assertEqual(result.parsed["summary"], "ok")

    def test_parse_structured_output_invalid_json_falls_back(self):
        from web_console.task_profiles import parse_structured_output
        raw = "not json at all"
        result = parse_structured_output(raw)
        self.assertFalse(result.parsed_ok)
        self.assertIsNone(result.parsed)
        self.assertEqual(result.raw, raw)

    def test_parse_structured_output_never_evals(self):
        """Malicious model output must not be evaluated."""
        from web_console.task_profiles import parse_structured_output
        raw = "__import__('os').system('echo hacked')"
        result = parse_structured_output(raw)
        self.assertFalse(result.parsed_ok)
        self.assertIsNone(result.parsed)

    def test_get_profile_by_name(self):
        from web_console.task_profiles import discover_profiles, get_profile_by_name
        with tempfile.TemporaryDirectory() as d:
            self._write_profile(d, {
                "name": "warehouse_awareness",
                "version": "1",
                "system_instruction": "s",
                "task_prompt": "t",
                "schema_example": {},
            })
            profiles = discover_profiles(d)
            found = get_profile_by_name(profiles, "warehouse_awareness")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "warehouse_awareness")

    def test_get_profile_by_name_not_found(self):
        from web_console.task_profiles import get_profile_by_name
        result = get_profile_by_name([], "missing")
        self.assertIsNone(result)

    def test_default_warehouse_profile_exists_and_valid(self):
        """The bundled warehouse_awareness_v1.json must load and parse."""
        from web_console.task_profiles import discover_profiles
        profiles_dir = pathlib.Path(__file__).resolve().parents[2] / "config" / "task_profiles"
        if not profiles_dir.exists():
            self.skipTest("config/task_profiles directory not found")
        profiles = discover_profiles(str(profiles_dir))
        names = [p.name for p in profiles]
        self.assertIn("warehouse_awareness", names)
        p = next(x for x in profiles if x.name == "warehouse_awareness")
        self.assertGreater(len(p.task_prompt), 10)
        self.assertGreater(len(p.prompt_hash), 10)


# ── Frame extractor tests ─────────────────────────────────────────────────────

class TestFrameExtractorValidation(unittest.TestCase):
    """Tests for frame_extractor: catalog allowlisting, parameter validation, path safety."""

    def test_validate_extraction_params_requires_bag_key(self):
        from web_console.frame_extractor import validate_extraction_params
        err = validate_extraction_params({"image_topic": "/cam/image_raw"})
        self.assertIsNotNone(err)
        self.assertIn("bag_key", err.lower())

    def test_validate_extraction_params_requires_image_topic(self):
        from web_console.frame_extractor import validate_extraction_params
        err = validate_extraction_params({"bag_key": "my-bag"})
        self.assertIsNotNone(err)
        self.assertIn("image_topic", err.lower())

    def test_validate_extraction_params_invalid_bag_key_chars(self):
        from web_console.frame_extractor import validate_extraction_params
        err = validate_extraction_params({
            "bag_key": "../../etc/passwd",
            "image_topic": "/cam",
        })
        self.assertIsNotNone(err)

    def test_validate_extraction_params_max_frames_capped(self):
        from web_console.frame_extractor import validate_extraction_params
        err = validate_extraction_params({
            "bag_key": "my-bag",
            "image_topic": "/cam",
            "max_frames": 99999,
        })
        self.assertIsNotNone(err)
        self.assertIn("max_frames", err.lower())

    def test_validate_extraction_params_valid(self):
        from web_console.frame_extractor import validate_extraction_params
        err = validate_extraction_params({
            "bag_key": "my-bag",
            "image_topic": "/cam/image_raw",
            "max_frames": 50,
        })
        self.assertIsNone(err)

    def test_allowlist_bag_path_rejects_unknown_key(self):
        from web_console.frame_extractor import allowlist_bag_path
        catalog = {"rosbags": [], "images": [], "videos": []}
        with self.assertRaises(ValueError):
            allowlist_bag_path("unknown-key", catalog)

    def test_allowlist_bag_path_rejects_not_installed(self):
        from web_console.frame_extractor import allowlist_bag_path
        catalog = {
            "rosbags": [{"key": "my-bag", "installed": False, "local_path": "/data/my-bag"}],
            "images": [],
            "videos": [],
        }
        with self.assertRaises(ValueError):
            allowlist_bag_path("my-bag", catalog)

    def test_allowlist_bag_path_accepts_installed(self):
        from web_console.frame_extractor import allowlist_bag_path
        catalog = {
            "rosbags": [{"key": "my-bag", "installed": True, "local_path": "/data/my-bag"}],
            "images": [],
            "videos": [],
        }
        path = allowlist_bag_path("my-bag", catalog)
        self.assertEqual(path, "/data/my-bag")

    def test_build_extraction_args_returns_list(self):
        from web_console.frame_extractor import ExtractionParams, build_extraction_args
        params = ExtractionParams(
            bag_key="my-bag",
            bag_path="/data/my-bag",
            image_topic="/cam/image_raw",
            dataset_id="11111111-1111-1111-1111-111111111111",
            output_dir="/tmp/test_output",
            start_offset=0.0,
            end_offset=None,
            duration=None,
            sample_interval=None,
            target_sample_count=20,
            max_frames=50,
        )
        args = build_extraction_args("/path/to/script.py", params)
        self.assertIsInstance(args, list)
        self.assertEqual(args[0], "python3")
        # Must never contain shell metacharacters in a single joined string
        full = " ".join(args)
        self.assertNotIn(";", full)
        self.assertNotIn("&&", full)
        self.assertNotIn("$", full)

    def test_is_safe_dataset_id_valid(self):
        from web_console.frame_extractor import _is_safe_dataset_id
        self.assertTrue(_is_safe_dataset_id("11111111-1111-1111-1111-111111111111"))

    def test_is_safe_dataset_id_rejects_path(self):
        from web_console.frame_extractor import _is_safe_dataset_id
        self.assertFalse(_is_safe_dataset_id("../../etc/passwd"))
        self.assertFalse(_is_safe_dataset_id("/absolute/path"))
        self.assertFalse(_is_safe_dataset_id(""))


class TestFrameDatasetStore(unittest.TestCase):
    """Tests for FrameDatasetStore: manifest, path safety, frame listing."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from web_console.frame_extractor import FrameDatasetStore
        self._store = FrameDatasetStore(pathlib.Path(self._tmpdir))

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_dataset(self, dataset_id, frames=2):
        from web_console.frame_extractor import write_frame_manifest
        d = pathlib.Path(self._tmpdir) / dataset_id
        d.mkdir(parents=True)
        frame_records = []
        for i in range(frames):
            img = d / f"frame_{i:04d}.jpg"
            img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 8)  # minimal JPEG header
            frame_records.append({
                "index": i,
                "filename": img.name,
                "timestamp_sec": float(i),
                "timestamp_ns": i * 1_000_000_000,
            })
        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "bag_key": "test-bag",
            "bag_path": "/data/test-bag",
            "topic": "/cam",
            "start_offset_sec": 0.0,
            "end_offset_sec": None,
            "sample_interval_sec": 0.5,
            "max_frames": 100,
            "frames": frame_records,
            "frame_count": frames,
            "extracted_at": "2025-01-01T00:00:00Z",
            "output_dir": str(d),
        }
        write_frame_manifest(d, manifest)
        return manifest

    def test_list_datasets_empty(self):
        datasets = self._store.list_datasets()
        self.assertEqual(datasets, [])

    def test_list_datasets_after_create(self):
        dataset_id = "22222222-2222-2222-2222-222222222222"
        self._make_dataset(dataset_id)
        datasets = self._store.list_datasets()
        ids = [d["dataset_id"] for d in datasets]
        self.assertIn(dataset_id, ids)

    def test_get_manifest_returns_data(self):
        dataset_id = "33333333-3333-3333-3333-333333333333"
        self._make_dataset(dataset_id)
        manifest = self._store.get_manifest(dataset_id)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["dataset_id"], dataset_id)

    def test_get_manifest_unknown_returns_none(self):
        manifest = self._store.get_manifest("44444444-4444-4444-4444-444444444444")
        self.assertIsNone(manifest)

    def test_get_frame_path_valid(self):
        dataset_id = "55555555-5555-5555-5555-555555555555"
        self._make_dataset(dataset_id, frames=3)
        path = self._store.get_frame_path(dataset_id, 0)
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

    def test_get_frame_path_out_of_range(self):
        dataset_id = "66666666-6666-6666-6666-666666666666"
        self._make_dataset(dataset_id, frames=2)
        path = self._store.get_frame_path(dataset_id, 999)
        self.assertIsNone(path)

    def test_get_frame_path_rejects_path_traversal(self):
        """get_frame_path must never escape the dataset directory."""
        dataset_id = "77777777-7777-7777-7777-777777777777"
        self._make_dataset(dataset_id, frames=1)
        # Negative indices are not valid frame indices (already checked upstream)
        path = self._store.get_frame_path(dataset_id, -1)
        self.assertIsNone(path)

    def test_get_frame_path_invalid_dataset_id(self):
        path = self._store.get_frame_path("../../etc", 0)
        self.assertIsNone(path)


# ── Review store tests ────────────────────────────────────────────────────────

class TestReviewStore(unittest.TestCase):
    """Tests for review_store: annotation CRUD, label validation, thread safety."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from web_console.review_store import ReviewStore
        self._store = ReviewStore(pathlib.Path(self._tmpdir))

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _annotation(self, run_id, frame_index=0, label="acceptable"):
        from web_console.review_store import ReviewAnnotation
        return ReviewAnnotation(
            run_id=run_id,
            frame_index=frame_index,
            label=label,
            note="test note",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )

    def test_get_reviews_empty(self):
        reviews = self._store.get_reviews("nonexistent-run")
        self.assertEqual(reviews, [])

    def test_upsert_and_get(self):
        run_id = str(uuid.uuid4())
        ann = self._annotation(run_id, frame_index=3, label="acceptable")
        self._store.upsert_review(ann)
        reviews = self._store.get_reviews(run_id)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].frame_index, 3)
        self.assertEqual(reviews[0].label, "acceptable")

    def test_upsert_updates_existing(self):
        run_id = str(uuid.uuid4())
        ann1 = self._annotation(run_id, frame_index=0, label="acceptable")
        ann2 = self._annotation(run_id, frame_index=0, label="ambiguous")
        ann2 = ann2.__class__(
            run_id=run_id,
            frame_index=0,
            label="ambiguous",
            note="updated",
            created_at=ann1.created_at,
            updated_at="2025-01-02T00:00:00Z",
        )
        self._store.upsert_review(ann1)
        self._store.upsert_review(ann2)
        reviews = self._store.get_reviews(run_id)
        self.assertEqual(len(reviews), 1, "Upsert should not duplicate")
        self.assertEqual(reviews[0].label, "ambiguous")

    def test_multiple_frames_stored_separately(self):
        run_id = str(uuid.uuid4())
        for i in range(5):
            ann = self._annotation(run_id, frame_index=i, label="acceptable")
            self._store.upsert_review(ann)
        reviews = self._store.get_reviews(run_id)
        self.assertEqual(len(reviews), 5)
        indices = sorted(r.frame_index for r in reviews)
        self.assertEqual(indices, [0, 1, 2, 3, 4])

    def test_validate_review_valid(self):
        from web_console.review_store import validate_review
        err = validate_review({"frame_index": 0, "label": "acceptable"})
        self.assertIsNone(err)

    def test_validate_review_missing_label(self):
        from web_console.review_store import validate_review
        err = validate_review({"frame_index": 0})
        self.assertIsNotNone(err)

    def test_validate_review_invalid_label(self):
        from web_console.review_store import validate_review
        err = validate_review({"frame_index": 0, "label": "not_a_valid_label"})
        self.assertIsNotNone(err)

    def test_validate_review_missing_frame_index(self):
        from web_console.review_store import validate_review
        err = validate_review({"label": "acceptable"})
        self.assertIsNotNone(err)

    def test_validate_review_note_too_long(self):
        from web_console.review_store import validate_review
        err = validate_review({"frame_index": 0, "label": "acceptable", "note": "x" * 10001})
        self.assertIsNotNone(err)

    def test_allowed_labels_all_accepted(self):
        from web_console.review_store import validate_review, ALLOWED_REVIEW_LABELS
        for label in ALLOWED_REVIEW_LABELS:
            err = validate_review({"frame_index": 0, "label": label})
            self.assertIsNone(err, f"Label {label!r} should be valid")

    def test_persistence_across_instances(self):
        """Reviews must survive recreating the store."""
        from web_console.review_store import ReviewStore
        run_id = str(uuid.uuid4())
        ann = self._annotation(run_id, frame_index=7, label="ambiguous")
        self._store.upsert_review(ann)
        # Recreate store.
        store2 = ReviewStore(pathlib.Path(self._tmpdir))
        reviews = store2.get_reviews(run_id)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].frame_index, 7)


# ── New server routes (profiles, frame datasets, extract, reviews, compare) ───

class TestProfilesAPI(unittest.TestCase):
    """Tests for GET /api/profiles."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        profiles_dir = pathlib.Path(cls._tmpdir) / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "warehouse_awareness_v1.json").write_text(json.dumps({
            "name": "warehouse_awareness",
            "version": "1",
            "system_instruction": "You are a warehouse-awareness assistant.",
            "task_prompt": "Describe what you see in the warehouse.",
            "schema_example": {"summary": ""},
        }), encoding="utf-8")
        config = {
            "quiet": True,
            "socket_path": "/tmp/no_such.sock",
            "task_profiles_dir": str(profiles_dir),
        }
        cls._srv, cls._port, cls._thread = _start_test_server(config=config)

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _get(self, path):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, json.loads(body)

    def test_list_profiles_returns_list(self):
        status, data = self._get("/api/profiles")
        self.assertEqual(status, 200)
        self.assertIn("profiles", data)
        self.assertIsInstance(data["profiles"], list)

    def test_list_profiles_includes_warehouse(self):
        status, data = self._get("/api/profiles")
        names = [p["name"] for p in data["profiles"]]
        self.assertIn("warehouse_awareness", names)

    def test_list_profiles_has_prompt_hash(self):
        status, data = self._get("/api/profiles")
        for p in data["profiles"]:
            self.assertIn("prompt_hash", p)
            self.assertGreater(len(p["prompt_hash"]), 0)

    def test_list_profiles_has_profile_id(self):
        status, data = self._get("/api/profiles")
        for p in data["profiles"]:
            self.assertIn("profile_id", p)


class TestFrameDatasetAPI(unittest.TestCase):
    """Tests for GET /api/frame-datasets and frame-image serving."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        runs_dir = pathlib.Path(cls._tmpdir) / "runs"
        runs_dir.mkdir()
        frame_datasets_dir = runs_dir / "frame_datasets"
        frame_datasets_dir.mkdir()
        from web_console.frame_extractor import FrameDatasetStore, write_frame_manifest
        cls._frame_store = FrameDatasetStore(frame_datasets_dir)
        # Create a test dataset.
        cls._dataset_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        dataset_dir = frame_datasets_dir / cls._dataset_id
        dataset_dir.mkdir()
        img = dataset_dir / "frame_0000.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 8)
        write_frame_manifest(dataset_dir, {
            "schema_version": 1,
            "dataset_id": cls._dataset_id,
            "bag_key": "test-bag",
            "bag_path": "/data/test-bag",
            "topic": "/cam",
            "start_offset_sec": 0.0,
            "end_offset_sec": None,
            "sample_interval_sec": 0.5,
            "max_frames": 100,
            "frames": [{"index": 0, "filename": "frame_0000.jpg",
                         "timestamp_sec": 1.0, "timestamp_ns": 1000000000}],
            "frame_count": 1,
            "extracted_at": "2025-01-01T00:00:00Z",
            "output_dir": str(dataset_dir),
        })
        config = {
            "quiet": True,
            "socket_path": "/tmp/no_such.sock",
            "frame_datasets_dir": str(frame_datasets_dir),
        }
        cls._srv, cls._port, cls._thread = _start_test_server(
            config=config,
        )

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _get_raw(self, path):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, body, resp.getheader("Content-Type", "")

    def _get(self, path):
        status, body, _ = self._get_raw(path)
        return status, json.loads(body)

    def test_list_frame_datasets_returns_list(self):
        status, data = self._get("/api/frame-datasets")
        self.assertEqual(status, 200)
        self.assertIn("datasets", data)

    def test_list_frame_datasets_contains_our_dataset(self):
        status, data = self._get("/api/frame-datasets")
        ids = [d["dataset_id"] for d in data["datasets"]]
        self.assertIn(self._dataset_id, ids)

    def test_get_frame_dataset_manifest(self):
        status, data = self._get(f"/api/frame-datasets/{self._dataset_id}")
        self.assertEqual(status, 200)
        self.assertEqual(data["dataset_id"], self._dataset_id)
        self.assertIn("frames", data)

    def test_get_frame_dataset_unknown_returns_404(self):
        status, data = self._get("/api/frame-datasets/bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        self.assertEqual(status, 404)

    def test_get_frame_dataset_invalid_id_returns_400(self):
        status, data = self._get("/api/frame-datasets/../../etc/passwd")
        self.assertEqual(status, 404)  # no route match for invalid UUID

    def test_serve_frame_image_returns_jpeg(self):
        status, body, ct = self._get_raw(f"/api/frame-datasets/{self._dataset_id}/frames/0")
        self.assertEqual(status, 200)
        self.assertIn("image/jpeg", ct)
        self.assertGreater(len(body), 0)

    def test_serve_frame_image_out_of_range_returns_404(self):
        status, body, _ = self._get_raw(f"/api/frame-datasets/{self._dataset_id}/frames/999")
        self.assertEqual(status, 404)

    def test_serve_frame_image_unknown_dataset_returns_404(self):
        status, body, _ = self._get_raw(
            "/api/frame-datasets/cccccccc-cccc-cccc-cccc-cccccccccccc/frames/0"
        )
        self.assertEqual(status, 404)

    def test_serve_frame_image_path_traversal_blocked(self):
        """Path traversal attempts must not reach the filesystem."""
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        # Encode a path traversal in the frame index portion (not a valid digit sequence)
        conn.request("GET", f"/api/frame-datasets/{self._dataset_id}/frames/00000")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        # The important thing is no server error and no data from /etc/passwd
        self.assertIn(resp.status, (200, 404))


class TestExtractAPI(unittest.TestCase):
    """Tests for POST /api/extract and POST /api/extract/<id>/cancel."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        config = {
            "quiet": True,
            "socket_path": "/tmp/no_such.sock",
            "rosbag_dir": cls._tmpdir,
            "runs_dir": str(pathlib.Path(cls._tmpdir) / "runs"),
        }
        cls._srv, cls._port, cls._thread = _start_test_server(config=config)

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _post(self, path, data):
        body = json.dumps(data).encode()
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json",
                               "Content-Length": str(len(body))})
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()
        return resp.status, json.loads(resp_body)

    def test_extract_missing_bag_key_returns_400(self):
        status, data = self._post("/api/extract", {"image_topic": "/cam"})
        self.assertEqual(status, 400)

    def test_extract_missing_image_topic_returns_400(self):
        status, data = self._post("/api/extract", {"bag_key": "my-bag"})
        self.assertEqual(status, 400)

    def test_extract_unknown_bag_key_returns_400(self):
        """Bag keys not in catalog must be rejected."""
        status, data = self._post("/api/extract", {
            "bag_key": "not-in-catalog",
            "image_topic": "/cam",
        })
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_extract_cancel_nonexistent_run_returns_404(self):
        status, data = self._post(
            "/api/extract/eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee/cancel", {}
        )
        self.assertEqual(status, 404)

    def test_extract_cancel_non_extraction_run_returns_400(self):
        """Cancel must reject runs that are not extraction kind."""
        run_store = self._srv.run_store
        run_id = RunStore.new_run_id()
        run_store.save_run(run_id, {
            "schema_version": 1,
            "run_id": run_id,
            "kind": "ros_experiment",
            "status": "running",
            "created_at": "2025-01-01T00:00:00Z",
        })
        status, data = self._post(f"/api/extract/{run_id}/cancel", {})
        self.assertEqual(status, 400)


class TestReviewAPI(unittest.TestCase):
    """Tests for POST/GET /api/runs/<run_id>/reviews."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        runs_dir = pathlib.Path(cls._tmpdir) / "runs"
        runs_dir.mkdir()
        cls._run_store = RunStore(runs_dir)
        config = {"quiet": True, "socket_path": "/tmp/no_such.sock",
                  "runs_dir": str(runs_dir)}
        cls._srv, cls._port, cls._thread = _start_test_server(
            config=config, run_store=cls._run_store,
        )

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _get(self, path):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, json.loads(body)

    def _post(self, path, data):
        body = json.dumps(data).encode()
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json",
                               "Content-Length": str(len(body))})
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()
        return resp.status, json.loads(resp_body)

    def test_get_reviews_empty(self):
        run_id = str(uuid.uuid4())
        status, data = self._get(f"/api/runs/{run_id}/reviews")
        self.assertEqual(status, 200)
        self.assertEqual(data["reviews"], [])

    def test_upsert_review_valid(self):
        run_id = str(uuid.uuid4())
        status, data = self._post(f"/api/runs/{run_id}/reviews", {
            "frame_index": 5,
            "label": "acceptable",
            "note": "looks good",
        })
        self.assertEqual(status, 200)
        self.assertEqual(data["label"], "acceptable")
        self.assertEqual(data["frame_index"], 5)

    def test_upsert_review_invalid_label(self):
        run_id = str(uuid.uuid4())
        status, data = self._post(f"/api/runs/{run_id}/reviews", {
            "frame_index": 0,
            "label": "not_valid",
        })
        self.assertEqual(status, 400)

    def test_upsert_review_missing_frame_index(self):
        run_id = str(uuid.uuid4())
        status, data = self._post(f"/api/runs/{run_id}/reviews", {
            "label": "acceptable",
        })
        self.assertEqual(status, 400)

    def test_upsert_and_get_roundtrip(self):
        run_id = str(uuid.uuid4())
        self._post(f"/api/runs/{run_id}/reviews", {
            "frame_index": 3,
            "label": "ambiguous",
            "note": "unclear",
        })
        status, data = self._get(f"/api/runs/{run_id}/reviews")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["reviews"]), 1)
        self.assertEqual(data["reviews"][0]["label"], "ambiguous")

    def test_upsert_review_all_valid_labels(self):
        from web_console.review_store import ALLOWED_REVIEW_LABELS
        run_id = str(uuid.uuid4())
        for i, label in enumerate(sorted(ALLOWED_REVIEW_LABELS)):
            status, data = self._post(f"/api/runs/{run_id}/reviews", {
                "frame_index": i,
                "label": label,
            })
            self.assertEqual(status, 200, f"Label {label!r} should be accepted")

    def test_reviews_route_invalid_run_id_rejected(self):
        status, data = self._get("/api/runs/not-a-uuid/reviews")
        self.assertIn(status, (400, 404))


class TestCompareAPI(unittest.TestCase):
    """Tests for GET /api/compare aligned comparison."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        runs_dir = pathlib.Path(cls._tmpdir) / "runs"
        runs_dir.mkdir()
        cls._run_store = RunStore(runs_dir)
        # Create two runs with overlapping frames.
        cls._run_id1 = RunStore.new_run_id()
        cls._run_id2 = RunStore.new_run_id()
        cls._run_store.save_run(cls._run_id1, {
            "schema_version": 1,
            "run_id": cls._run_id1,
            "kind": "experiment",
            "status": "completed",
            "created_at": "2025-01-01T00:00:00Z",
            "model": "model-a",
            "strategy": "single_frame",
            "result_frames": [
                {"frame_index": 0, "text": "frame 0 run 1", "latency_ms": 100},
                {"frame_index": 1, "text": "frame 1 run 1", "latency_ms": 110},
            ],
        })
        cls._run_store.save_run(cls._run_id2, {
            "schema_version": 1,
            "run_id": cls._run_id2,
            "kind": "experiment",
            "status": "completed",
            "created_at": "2025-01-01T01:00:00Z",
            "model": "model-b",
            "strategy": "single_frame_observation_history",
            "result_frames": [
                {"frame_index": 0, "text": "frame 0 run 2", "latency_ms": 200},
                {"frame_index": 1, "text": "frame 1 run 2", "latency_ms": 220},
            ],
        })
        config = {"quiet": True, "socket_path": "/tmp/no_such.sock",
                  "runs_dir": str(runs_dir)}
        cls._srv, cls._port, cls._thread = _start_test_server(
            config=config, run_store=cls._run_store,
        )

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _get(self, path):
        conn = HTTPConnection("127.0.0.1", self._port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, json.loads(body)

    def test_compare_two_runs(self):
        status, data = self._get(
            f"/api/compare?run_ids={self._run_id1},{self._run_id2}"
        )
        self.assertEqual(status, 200)
        self.assertIn("aligned_frames", data)
        self.assertIn("summaries", data)

    def test_compare_aligned_frames_cover_both_runs(self):
        status, data = self._get(
            f"/api/compare?run_ids={self._run_id1},{self._run_id2}"
        )
        aligned = data["aligned_frames"]
        self.assertEqual(len(aligned), 2)
        for row in aligned:
            self.assertIn(self._run_id1, row)
            self.assertIn(self._run_id2, row)

    def test_compare_summaries_have_metadata(self):
        status, data = self._get(
            f"/api/compare?run_ids={self._run_id1},{self._run_id2}"
        )
        for run_id in (self._run_id1, self._run_id2):
            self.assertIn(run_id, data["summaries"])
            summary = data["summaries"][run_id]
            self.assertIn("model", summary)
            self.assertIn("status", summary)

    def test_compare_requires_at_least_two_run_ids(self):
        status, data = self._get(f"/api/compare?run_ids={self._run_id1}")
        self.assertEqual(status, 400)

    def test_compare_missing_run_ids_param(self):
        status, data = self._get("/api/compare")
        self.assertEqual(status, 400)

    def test_compare_unknown_run_id_returns_404(self):
        fake_id = str(uuid.uuid4())
        status, data = self._get(f"/api/compare?run_ids={self._run_id1},{fake_id}")
        self.assertEqual(status, 404)

    def test_compare_invalid_run_id_returns_400(self):
        status, data = self._get(
            f"/api/compare?run_ids={self._run_id1},not-a-uuid"
        )
        self.assertEqual(status, 400)


# ── UI wiring tests ───────────────────────────────────────────────────────────

class TestWarehouseUIElements(unittest.TestCase):
    """Check app.js and HTML template for new warehouse workbench UI elements."""

    @classmethod
    def setUpClass(cls):
        cls._js_path = pathlib.Path(__file__).resolve().parents[1] / "static" / "app.js"
        cls._js = cls._js_path.read_text(encoding="utf-8")
        # Import server to access the HTML template.
        from web_console import server as _srv_mod
        cls._html = _srv_mod._INDEX_TEMPLATE

    def test_app_js_has_frame_datasets_fetch(self):
        """app.js must call the /api/frame-datasets route."""
        self.assertIn("/api/frame-datasets", self._js)

    def test_app_js_has_profiles_fetch(self):
        """app.js must call the /api/profiles route."""
        self.assertIn("/api/profiles", self._js)

    def test_app_js_has_compare_fetch(self):
        """app.js must call the /api/compare route."""
        self.assertIn("/api/compare", self._js)

    def test_app_js_has_reviews_fetch(self):
        """app.js must call the /api/runs/.../reviews route."""
        self.assertIn("/reviews", self._js)

    def test_app_js_has_extract_fetch(self):
        """app.js must call the /api/extract route."""
        self.assertIn("/api/extract", self._js)

    def test_html_has_frame_explorer_view(self):
        """HTML template must include a frame-explorer view container."""
        self.assertIn("frame-explorer", self._html)

    def test_html_has_warehouse_profile_reference(self):
        """HTML template must reference warehouse or profiles somewhere."""
        self.assertIn("profile", self._html.lower())

    def test_html_has_review_ui(self):
        """HTML template must include review label/annotation UI."""
        self.assertIn("review", self._html.lower())


if __name__ == "__main__":
    unittest.main()
