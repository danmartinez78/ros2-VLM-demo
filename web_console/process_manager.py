# Copyright 2025 edge_vlm_ros contributors
"""
Subprocess and process-group lifecycle management for the web console.

Safety rules enforced here:
- Never use shell=True.
- Track only process groups started by this manager.
- Stop owned process groups with SIGTERM → bounded wait → SIGKILL.
- Reject concurrent conflicting ROS experiments.
- Apply output-size bounds to log collection.
- Clean up all owned children on shutdown.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_MAX_LOG_LINES: int = 1000
_TERM_WAIT_SECONDS: float = 5.0
_KILL_WAIT_SECONDS: float = 3.0

# Completion callback type: (run_id, exit_code, was_explicitly_stopped, log_lines)
CompletionCallback = Optional[Callable[[str, Optional[int], bool, List[str]], None]]


@dataclass
class _ManagedProcess:
    run_id: str
    kind: str  # "ros" or "infer"
    proc: subprocess.Popen
    pgid: int
    log_lines: List[str] = field(default_factory=list)
    stopped: bool = False
    exit_code: Optional[int] = None
    on_complete: Any = None  # CompletionCallback
    _log_done: threading.Event = field(default_factory=threading.Event)
    _callback_fired: bool = False
    _explicitly_stopped: bool = False


class ProcessManager:
    """Thread-safe manager for subprocesses started by the web console."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: Dict[str, _ManagedProcess] = {}
        # Slot reserved atomically during start_ros_experiment so concurrent
        # callers are rejected before Popen is even called.
        self._reserved_ros_run_id: Optional[str] = None

    # ── public API ────────────────────────────────────────────────────────────

    def start_ros_experiment(
        self,
        run_id: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        on_complete: CompletionCallback = None,
    ) -> int:
        """Start a ROS experiment subprocess in a new session.

        Returns the PID of the launched process.
        Raises RuntimeError if a ROS experiment is already running.
        The check and slot reservation are atomic — concurrent calls are
        rejected even before Popen is invoked.
        Never uses shell=True.
        """
        with self._lock:
            active = [
                p
                for p in self._processes.values()
                if not p.stopped and p.kind == "ros"
            ]
            if active or self._reserved_ros_run_id is not None:
                active_id = active[0].run_id if active else self._reserved_ros_run_id
                raise RuntimeError(
                    f"A ROS experiment is already running (run_id={active_id!r}). "
                    "Stop it before starting a new one."
                )
            # Atomically reserve the slot; released after registration or on error.
            self._reserved_ros_run_id = run_id

        try:
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
                close_fds=True,
            )
            pgid = os.getpgid(proc.pid)
            mp = _ManagedProcess(
                run_id=run_id,
                kind="ros",
                proc=proc,
                pgid=pgid,
                on_complete=on_complete,
            )
            with self._lock:
                self._processes[run_id] = mp
                self._reserved_ros_run_id = None
            threading.Thread(target=self._collect_logs, args=(mp,), daemon=True).start()
            threading.Thread(target=self._wait_proc, args=(mp,), daemon=True).start()
            return proc.pid
        except Exception:
            with self._lock:
                self._reserved_ros_run_id = None
            raise

    def stop_experiment(self, run_id: str) -> None:
        """Stop a managed process by run_id.

        Raises KeyError if the run_id is not found.
        """
        with self._lock:
            mp = self._processes.get(run_id)
        if mp is None:
            raise KeyError(f"No managed process with run_id={run_id!r}")
        self._stop_process(mp)

    def stop_all_ros(self) -> None:
        """Stop all running ROS experiments."""
        with self._lock:
            active = [
                p for p in self._processes.values() if not p.stopped and p.kind == "ros"
            ]
        for mp in active:
            self._stop_process(mp)

    def is_running(self, run_id: str) -> bool:
        with self._lock:
            mp = self._processes.get(run_id)
            if mp is None:
                return False
            return not mp.stopped and mp.proc.poll() is None

    def get_logs(self, run_id: str) -> List[str]:
        with self._lock:
            mp = self._processes.get(run_id)
            if mp is None:
                return []
            return list(mp.log_lines)

    def get_exit_code(self, run_id: str) -> Optional[int]:
        with self._lock:
            mp = self._processes.get(run_id)
            if mp is None:
                return None
            return mp.exit_code

    def active_ros_run_id(self) -> Optional[str]:
        with self._lock:
            if self._reserved_ros_run_id is not None:
                return self._reserved_ros_run_id
            for mp in self._processes.values():
                if not mp.stopped and mp.kind == "ros" and mp.proc.poll() is None:
                    return mp.run_id
            return None

    def cleanup(self) -> None:
        """Stop all owned processes; call on application shutdown."""
        with self._lock:
            all_procs = list(self._processes.values())
        for mp in all_procs:
            if not mp.stopped:
                self._stop_process(mp)

    # ── internal ──────────────────────────────────────────────────────────────

    def _stop_process(self, mp: _ManagedProcess) -> None:
        if mp.stopped:
            return
        # Mark as explicitly stopped before sending signals so _fire_completion
        # (which may fire from _wait_proc concurrently) reads the correct flag.
        with self._lock:
            mp._explicitly_stopped = True
        try:
            os.killpg(mp.pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        deadline = time.monotonic() + _TERM_WAIT_SECONDS
        while time.monotonic() < deadline:
            if mp.proc.poll() is not None:
                break
            time.sleep(0.05)

        if mp.proc.poll() is None:
            try:
                os.killpg(mp.pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                mp.proc.wait(timeout=_KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass

        with self._lock:
            mp.stopped = True
            mp.exit_code = mp.proc.poll()
        self._fire_completion(mp)

    def _collect_logs(self, mp: _ManagedProcess) -> None:
        if mp.proc.stdout is None:
            mp._log_done.set()
            return
        try:
            for raw_line in mp.proc.stdout:
                line = raw_line.decode(errors="replace").rstrip("\n\r")
                with self._lock:
                    if len(mp.log_lines) < _MAX_LOG_LINES:
                        mp.log_lines.append(line)
        finally:
            try:
                mp.proc.stdout.close()
            except OSError:
                pass
            mp._log_done.set()

    def _wait_proc(self, mp: _ManagedProcess) -> None:
        mp.proc.wait()
        with self._lock:
            mp.stopped = True
            mp.exit_code = mp.proc.returncode
        self._fire_completion(mp)

    def _fire_completion(self, mp: _ManagedProcess) -> None:
        """Fire the on_complete callback exactly once, after log collection finishes."""
        # Wait for _collect_logs to drain stdout (max 2 s).
        mp._log_done.wait(timeout=2.0)
        with self._lock:
            if mp._callback_fired:
                return
            mp._callback_fired = True
            exit_code = mp.exit_code
            logs = list(mp.log_lines)
            was_stopped = mp._explicitly_stopped
        if mp.on_complete is not None:
            try:
                mp.on_complete(mp.run_id, exit_code, was_stopped, logs)
            except Exception:
                pass
