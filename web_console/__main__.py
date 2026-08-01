# Copyright 2025 edge_vlm_ros contributors
"""Entry point: python -m web_console [--host H] [--port P] [--socket PATH]"""
from __future__ import annotations

import argparse
import signal
import sys
import pathlib

from .server import ConsoleServer, DEFAULT_HOST, DEFAULT_PORT


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="web_console",
        description=(
            "Local web experiment console for edge_vlm_ros. "
            "Binds to 127.0.0.1 by default."
        ),
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port (default: 8765)")
    parser.add_argument(
        "--socket",
        default="/tmp/edge_vlm.sock",
        help="Path to edge_vlm_server IPC socket (default: /tmp/edge_vlm.sock)",
    )
    parser.add_argument(
        "--cli",
        default="edge_vlm_cli",
        help="Path to edge_vlm_cli binary (default: edge_vlm_cli on PATH)",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(pathlib.Path.home() / ".web_console" / "runs"),
        help="Directory to store run records",
    )
    parser.add_argument(
        "--ros-script",
        default=str(
            pathlib.Path(__file__).parent.parent
            / "scripts"
            / "test_data"
            / "run_image_proc_test.sh"
        ),
        help="Path to the ROS experiment shell script",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress request logging")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    config = {
        "socket_path": args.socket,
        "cli_path": args.cli,
        "runs_dir": args.runs_dir,
        "ros_script_path": args.ros_script,
        "quiet": args.quiet,
    }
    srv = ConsoleServer(host=args.host, port=args.port, config=config)

    def _shutdown(sig, frame):
        print("\n[web_console] Shutting down…", flush=True)
        t = __import__("threading").Thread(target=srv.shutdown_gracefully, daemon=True)
        t.start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[web_console] Listening on http://{args.host}:{args.port}/", flush=True)
    print(f"[web_console] IPC socket: {args.socket}", flush=True)
    print("[web_console] Press Ctrl+C to stop.", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
