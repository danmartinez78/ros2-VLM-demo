# Web Experiment Console

A lightweight local HTTP control plane for running and inspecting standalone
Edge-VLM inference and bounded ROS/rosbag experiments without a remote desktop
or repeated terminal orchestration.

**This console is an experiment control plane.** It does not embed TensorRT
inference, replace the standalone `edge_vlm_server`, or participate in the
frame-to-inference data path.

## Installation and launch

The console uses only Python 3 standard library — no pip install is needed.

```bash
# From the repository root:
python3 -m web_console
```

Default options:

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8765` | TCP port |
| `--socket` | `/tmp/edge_vlm.sock` | Path to `edge_vlm_server` IPC socket |
| `--cli` | `edge_vlm_cli` | Path to `edge_vlm_cli` binary |
| `--runs-dir` | `~/.web_console/runs` | Directory for run records |
| `--ros-script` | `scripts/test_data/run_image_proc_test.sh` | ROS experiment script |
| `--quiet` | off | Suppress request logging |

Once started, open **http://127.0.0.1:8765/** in your browser.

### Example with explicit paths

```bash
source scripts/edge_vlm_env.sh
source "$ROS_WORKSPACE/install/setup.bash"

python3 -m web_console \
  --socket "$WORKER_SOCKET_PATH" \
  --cli "$ROS_WORKSPACE/install/edge_vlm_ros/lib/edge_vlm_ros/edge_vlm_cli" \
  --runs-dir "$HOME/.web_console/runs"
```

## Remote access

### SSH port forwarding

```bash
# On your laptop — forward local port 8765 to the Jetson:
ssh -L 8765:127.0.0.1:8765 user@jetson-host

# Then open http://localhost:8765/ locally.
```

### Tailscale

If the Jetson is enrolled in your Tailscale network, start the console with:

```bash
python3 -m web_console --host 100.x.y.z --port 8765
```

Replace `100.x.y.z` with the Jetson's Tailscale IP. Access from any device on
the same tailnet at `http://100.x.y.z:8765/`.

> **Trust boundary:** The console is designed for single-user, localhost-only
> use. There is no authentication. Do not expose it on a public or shared
> network interface.

### Trusted-LAN access

For direct browser access from another machine on the same private network
(e.g. a laptop on the same lab switch as the Jetson), bind to all interfaces:

```bash
python3 -m web_console --host 0.0.0.0 --port 8765
# Then open http://<thor-lan-ip>:8765/ from the other machine.
```

When the configured host is not a loopback address, the console prints a
conspicuous warning at startup reminding you that the API is unauthenticated.

**Firewall guidance (Ubuntu / Jetson):**

```bash
# Allow access only from the trusted LAN subnet (replace with your subnet):
sudo ufw allow from 192.168.1.0/24 to any port 8765 proto tcp
sudo ufw deny 8765
sudo ufw reload
```

> ⚠️ **Do not** open port 8765 through a router NAT rule or expose it on a
> public or shared network interface. There is no authentication — any reachable
> client can trigger inference requests and control ROS experiment processes.
> `127.0.0.1` remains the recommended default for all single-machine use.

## Architecture and trust boundary

```
Browser ──HTTP──► ConsoleServer (127.0.0.1:8765)
                      │
                      ├─► status_collector   (read-only: socket connect probe, ss, nvidia-smi)
                      ├─► inference_client   (subprocess: edge_vlm_cli — arg list, no shell)
                      ├─► ProcessManager     (subprocess: bash <ros_script> — arg list, no shell)
                      └─► RunStore           (local JSON manifests under ~/.web_console/runs/)

ConsoleServer ─IPC─► edge_vlm_server        (pre-existing worker; not started/stopped by console)
```

The console:
- Never starts or stops `edge_vlm_server` automatically.
- Never accepts arbitrary shell commands or executable paths from HTTP requests.
- Constructs all subprocess argument arrays directly (no `shell=True`).
- Tracks and stops only process groups that it started.
- Validates run IDs against a UUID pattern to prevent path traversal.
- Rejects concurrent conflicting ROS experiments with HTTP 409.
- Applies upload size limits (64 MiB) and allowed image extension lists.
- Degrades cleanly on CPU-only/CI systems (no CUDA, TensorRT, or nvidia-smi required).

## Supported MVP workflows

### Status

The **Status** section shows:
- Whether `edge_vlm_server` is reachable on the configured socket.
- GPU utilisation and memory (when `nvidia-smi` is available).
- Relevant environment variables.
- Active ROS experiment run ID (if any).

### Standalone inference

1. Select a supported image file (JPEG, PNG, BMP, WebP, TIFF — max 64 MiB).
2. Enter a prompt.
3. Adjust optional parameters (max tokens, temperature, top-p, top-k).
4. Click **Run Inference**.

The console calls `edge_vlm_cli` through the IPC socket. The full response,
inference duration, and a downloadable JSON manifest are preserved in the run
history.

### Bounded ROS experiment

1. Fill in the allowlisted parameters (image topic, prompt, delivery mode,
   observation history, playback duration, timeouts, required results).
2. Click **Start ROS Experiment**.

The console launches `scripts/test_data/run_image_proc_test.sh` as an isolated
subprocess. Live log lines are polled and displayed. Click **Stop** to send
SIGTERM (with SIGKILL fallback) to the experiment process group.

**Only the experiment started by the console is stopped.** Pre-existing
`edge_vlm_ros_node` and `edge_vlm_server` processes are never touched.

### Run history

All run manifests are stored under `--runs-dir`. Click a run in the history
table to inspect its full configuration, result text, latency, and error
information.

### Sequence Catalog (Frame Explorer)

The **Sequence Catalog** panel in Frame Explorer provides access to locally
installed datasets without a live ROS graph.  Three adapter types are
supported:

| Adapter | Dataset | Source |
|---|---|---|
| `ros_static_fixture` | RT-DETR quickstart rosbag | A rosbag with exactly one raw image message |
| `nuscenes_scene` | nuScenes mini | `CAM_FRONT` keyframes from the `scene → sample` linked list |
| `jaad_clip` | JAAD | Individual `video_XXXX.mp4` clips with annotation summaries |

**Browsing is lazy and bounded.**  The catalog response is sequence-level only
— frame references are not pre-loaded for JAAD clips at discovery time.  The
UI shows a compact dataset/source selector, a searchable sequence list with
paginated results (20 visible rows), and a selected-sequence detail panel.
A ≤ 20-thumbnail strip is rendered for sequences that have pre-indexed frame
refs (nuScenes, static fixtures); JAAD clips show a sampled navigator with a
direct frame-index input.

**Frame serving and media backends.**  On-demand frame extraction uses two
backends with automatic fallback:

1. **ffmpeg / ffprobe** — preferred when available on `PATH`.
2. **OpenCV (`cv2`)** — used automatically when ffmpeg/ffprobe are absent
   (e.g. on a fresh Thor without media tools).  Install with
   `pip install opencv-python-headless`.

`GET /api/sequences` includes a `decoder_capability` object that reports which
backends are active and surfaces an actionable error when neither is available:

```json
{
  "ffprobe": false,
  "ffmpeg": false,
  "opencv": true,
  "metadata_probe": "opencv",
  "frame_extraction": "opencv",
  "actionable_error": null
}
```

Install at least one backend before using JAAD frame viewing:

```bash
# Option A — ffmpeg (also covers ffprobe):
sudo apt-get install ffmpeg

# Option B — OpenCV headless (no display required):
pip install opencv-python-headless
```

**RT-DETR static fixture prerequisite.**  The `ros_static_fixture` adapter
lists the RT-DETR quickstart bag as a selectable sequence, but frame serving
requires the bag to have been previously extracted through the normal Frame
Explorer extraction flow.  The "Use in Experiment" button and frame viewer are
enabled only after an extraction has populated the frame dataset store.  Use
the existing **Extract Frames** panel to extract the bag first.

**Sequence experiments.**  Clicking **Use in Experiment** in the sequence
detail panel transfers the selected `dataset_id`, `sequence_id`, and optional
frame indices to the Experiment form.  `POST /api/sequences/experiment` runs
`ExperimentDefinition` on the materialized frames and records
`adapter`/`dataset_id`/`sequence_id`/`frame_source_ids` in the run manifest.
Lazy JAAD sequences require explicit `frame_indices` in the request body.

Configure dataset roots via server config keys or environment variables:

| Config key | Env variable | Default |
|---|---|---|
| `nuscenes_dir` | `NUSCENES_DIR` | `test_data/datasets/nuscenes-mini` |
| `jaad_dir` | `JAAD_DIR` | `test_data/datasets/jaad` |



## Artifact locations and retention

Each run is stored under `~/.web_console/runs/<run_id>/` (configurable with
`--runs-dir`).  The console exclusively owns `manifest.json` in that directory;
the ROS experiment script writes its outputs into the nested `artifacts/`
subdirectory so the two manifests never collide.

| Path | Owner | Contents |
|---|---|---|
| `…/<run_id>/manifest.json` | web-console | Full run record: config, params, status, lifecycle timestamps, artifact paths, parsed script manifest |
| `…/<run_id>/artifacts/manifest.json` | `run_image_proc_test.sh` | Script run manifest (git commit, bag path, engine dirs, result counts) |
| `…/<run_id>/artifacts/benchmark.jsonl` | `run_image_proc_test.sh` | Per-frame inference latency (one JSON object per line) |
| `…/<run_id>/artifacts/launch.log` | `run_image_proc_test.sh` | `ros2 launch` stdout/stderr |
| `…/<run_id>/artifacts/results.log` | `run_image_proc_test.sh` | `ros2 topic echo` output for `/vlm/result` |

When the experiment finishes, the console manifest is updated with:
- `"artifacts"` — list of safe relative paths (e.g. `["artifacts/manifest.json", "artifacts/benchmark.jsonl", …]`)
- `"script_manifest"` — parsed contents of `artifacts/manifest.json` (summary of the script run)

Log file contents are not embedded in the console manifest.  The
`GET /api/runs/<run_id>/logs` endpoint exposes the bounded in-memory log captured
during the run.

The 100 most recent runs are retained; older directories are removed
automatically when a new run is saved.

## Clean shutdown and recovery

Press **Ctrl+C** or send `SIGTERM` to the console process. It will:
1. Send SIGTERM to all owned process groups.
2. Wait up to 5 s for each group to exit.
3. Send SIGKILL to any remaining processes.
4. Stop the HTTP server.

If the console exits unexpectedly, any ROS experiment it launched will continue
running in its own process group. Identify and stop it with:

```bash
# Find running edge_vlm_ros processes:
pgrep -a -f edge_vlm_ros_node
pgrep -a -f edge_vlm_server

# Stop by process group (replace PGID with the value from ps):
kill -TERM -- -<PGID>
sleep 3
kill -KILL -- -<PGID> 2>/dev/null || true
```

Stale IPC sockets can be removed with `rm -f /tmp/edge_vlm.sock`.

## Thor validation checklist

After cloud CI passes, manually verify on a prepared Thor:

1. Start the standalone `edge_vlm_server`:
   ```bash
   source scripts/edge_vlm_env.sh
   edge_vlm_server "$EDGE_VLM_LLM_ENGINE_DIR" "$EDGE_VLM_MULTIMODAL_ENGINE_DIR" \
     "$EDGELLM_PLUGIN_PATH" /tmp/edge_vlm.sock 90 60 &
   ```
2. Start the web console: `python3 -m web_console`
3. Open `http://127.0.0.1:8765/` and verify the status badge shows **server reachable**.
4. Submit at least two image inference requests from the UI and confirm:
   - Both return successful responses.
   - The server PID shown in the status JSON does not change between requests.
5. Start a ROS experiment (image-proc rosbag workflow) from the UI and wait for
   a successful `/vlm/result` output in the live log.
6. Click **Stop** for the ROS experiment. Confirm:
   - The standalone server status badge remains green.
   - `edge_vlm_cli` runs successfully from a separate terminal.
7. Run `edge_vlm_cli` successfully after stopping the ROS experiment.
8. Confirm no orphan UI-owned processes or temporary sockets remain:
   ```bash
   pgrep -a -f edge_vlm_ros_node || echo "none"
   ls /tmp/web_console_* 2>/dev/null || echo "none"
   ```
