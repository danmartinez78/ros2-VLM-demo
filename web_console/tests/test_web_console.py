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
        """Artifacts written to $ARTIFACT_DIR are included in the finalized manifest."""
        script = (
            "#!/bin/bash\n"
            'mkdir -p "$ARTIFACT_DIR"\n'
            'echo \'{"score": 1.0, "pass": true}\' > "$ARTIFACT_DIR/result.json"\n'
            "exit 0\n"
        )
        status, data = self._start_ros(script)
        self.assertEqual(status, 202)
        run_id = data["run_id"]
        # Verify ARTIFACT_DIR was set to the run's directory in the store.
        self.assertIn("artifact_dir", data)

        record = self._poll_terminal(run_id, timeout=10.0)
        self.assertEqual(record.get("status"), "completed")
        artifacts = record.get("artifacts", {})
        self.assertIn(
            "result.json", artifacts,
            "result.json written to ARTIFACT_DIR must appear in finalized manifest",
        )
        self.assertAlmostEqual(artifacts["result.json"].get("score"), 1.0)
        self.assertTrue(artifacts["result.json"].get("pass"))

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


if __name__ == "__main__":
    unittest.main()
