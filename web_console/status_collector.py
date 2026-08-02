# Copyright 2025 edge_vlm_ros contributors
"""
Reads edge_vlm_server reachability, PID, and GPU status.

All collection is read-only and uses bounded subprocesses.
Degrades cleanly when nvidia-smi, ss, or the IPC socket are unavailable.
"""
from __future__ import annotations

import os
import re
import socket as _socket
import subprocess
from typing import Any, Dict, Optional


def check_server_reachable(socket_path: str) -> Dict[str, Any]:
    """Try connecting to the IPC Unix-domain socket and return a status dict.

    Returns::

        {"reachable": bool, "socket_path": str, "error"?: str}
    """
    result: Dict[str, Any] = {"reachable": False, "socket_path": socket_path}
    if not socket_path:
        result["error"] = "socket_path not configured"
        return result
    if not os.path.exists(socket_path):
        result["error"] = "socket file not found"
        return result
    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(socket_path)
        result["reachable"] = True
    except OSError as exc:
        result["error"] = str(exc)
    return result


def get_server_pid(socket_path: str) -> Optional[int]:
    """Return the PID listening on *socket_path* using bounded ss output."""
    if not socket_path or not os.path.exists(socket_path):
        return None
    try:
        result = subprocess.run(
            ["ss", "-lxnp"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        for line in result.stdout.splitlines():
            if socket_path not in line:
                continue
            match = re.search(r"pid=(\d+)", line)
            if match:
                return int(match.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def get_gpu_status() -> Dict[str, Any]:
    """Run nvidia-smi and return bounded GPU information.

    Returns::

        {"available": True, "gpus": [...]}
        or
        {"available": False, "error": str}
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"available": False, "error": result.stderr.strip() or "nvidia-smi error"}
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "utilization_pct": parts[2],
                        "memory_used_mib": parts[3],
                        "memory_total_mib": parts[4],
                    }
                )
        return {"available": True, "gpus": gpus}
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timed out"}
    except OSError as exc:
        return {"available": False, "error": str(exc)}


def collect_status(socket_path: str) -> Dict[str, Any]:
    """Collect all status information in one call."""
    server = check_server_reachable(socket_path)
    pid = get_server_pid(socket_path) if server["reachable"] else None
    gpu = get_gpu_status()
    env_vars = {
        k: os.environ.get(k, "")
        for k in (
            "EDGE_VLM_LLM_ENGINE_DIR",
            "EDGE_VLM_MULTIMODAL_ENGINE_DIR",
            "EDGELLM_PLUGIN_PATH",
            "WORKER_SOCKET_PATH",
            "ROS_DISTRO",
        )
    }
    return {
        "server": server,
        "server_pid": pid,
        "gpu": gpu,
        "env": env_vars,
    }
