# Jetson AGX Thor TensorRT Edge-LLM prefill stall RCA

Status: resolved by production process isolation  
Opened: 2026-07-30  
Validated platform: NVIDIA Jetson AGX Thor, JetPack 7.2 / R39.2

## Executive summary

The original ROS 2 Cosmos integration initialized successfully but stalled
indefinitely during TensorRT base-model prefill after processing its first
image. NVIDIA's native `llm_inference` executable completed normally with the
same engine and extracted image.

Two independent integration problems were found:

1. stale CUDA device code built for `sm_75` caused an initial
   `device kernel image is invalid` failure on Thor (`sm_110a`);
2. after correcting the CUDA architecture, co-loading the ROS 2 transitive
   native dependency set and TensorRT Edge-LLM in one process reproducibly
   triggered an indefinitely active Blackwell fused-attention prefill kernel.

The supported fix separates the system into two native processes:

```text
edge_vlm_ros_node          ROS 2 only
        <-> bounded Unix-domain socket
edge_vlm_server  ROS-free, direct-linked TensorRT Edge-LLM
```

This architecture completed repeated rosbag inference, retained in-memory
image loading, and recovered after the worker was killed and respawned. The
exact ROS transitive library or symbol interaction that triggers the in-process
stall has not been identified. Process isolation is the proven remedy; a more
specific low-level root-cause claim would be speculation.

## Tested system

| Component | Observed value |
| --- | --- |
| Device | NVIDIA Jetson AGX Thor |
| CPU architecture | aarch64 |
| GPU compute capability | 11.0 |
| Jetson Linux / JetPack | R39.2 / 7.2 |
| CUDA / driver | 13.2 / 595.78 |
| ROS 2 | Jazzy |
| Isaac ROS | 4.5 tooling at incident time (current JP7.2 bootstrap targets Isaac ROS 4.6) |
| Model | Cosmos-Reason2-8B, NVFP4 |
| Engine limits | 1024-token input, 4096-token KV capacity |

Representative artifact layout:

```text
$HOME/TensorRT-Edge-LLM
$HOME/TensorRT-Edge-LLM/build/cpp/libedgellmCore.a
$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
$HOME/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engine
$HOME/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engine/llm
$HOME/ros2_ws
```

## Failure 1: incorrect CUDA architecture

The first runtime construction attempts failed with:

```text
CUDA runtime error in cudaOccupancyMaxActiveBlocksPerMultiprocessor(...):
device kernel image is invalid
```

Artifact inspection showed:

```text
Thor compute capability: 11.0
Edge-LLM RoPE object:     sm_110a
Initial ROS consumer:     sm_75
```

Rebuilding Edge-LLM and the final consumer for `sm_110a` fixed runtime
construction. Tensor allocation, tokenizer loading, visual-engine loading, and
CUDA graph capture then completed. This was required but did not fix inference.

## Failure 2: fused-attention prefill stall

After the first image arrived, the in-process integration stopped after:

```text
Processing vision inputs
Switching optimization profile from: 1 to 0
```

GPU utilization remained near 98 percent for more than a minute. CUDA-GDB
showed the host waiting in:

```text
cudaStreamSynchronize
LLMInferenceRuntime::runBaseModelPrefill
LLMInferenceRuntime::handleRequest
TensorRTEdgeLLMBackend::infer
```

The active GPU kernel was a TensorRT Blackwell fused multi-head attention
forward kernel:

```text
grid:  (22, 1, 1)
block: (512, 1, 1)
state: active indefinitely
```

No contemporaneous NVRM Xid, MMU fault, or channel exception was found.

## Input and engine controls

The NVIDIA image-proc rosbag frame was validated as:

```text
width:        1920
height:       1200
encoding:     rgb8
step:         5760
payload size: 6912000 bytes
```

The row step equals `width * 3`; there was no unexpected padding or truncated
payload. The exact frame was extracted and resized to 512x320.

NVIDIA's native executable completed in approximately 1.6 seconds and returned
a coherent warehouse description:

```bash
./build/examples/llm/llm_inference \
  --engineDir "$WORKSPACE_DIR/$MODEL_NAME/engine/llm" \
  --multimodalEngineDir "$WORKSPACE_DIR/$MODEL_NAME/engine" \
  --inputFile /tmp/input_hawk.json \
  --outputFile /tmp/output_hawk.json \
  --maxGenerateLength 64 \
  --dumpOutput
```

This validated the engine bundle, visual engine, tokenizer, chat template,
plugin, image, CUDA runtime, TensorRT runtime, and rebuilt `sm_110a` code.

## Experiment matrix

| Experiment | Result | What it established |
| --- | --- | --- |
| Rebuild final CUDA consumer for `sm_110a` | Initialization succeeds | Architecture mismatch fixed |
| Reduce image width from 1280 to 512 | Stalls | Resolution not causal |
| Native `llm_inference` with exact frame | Succeeds | Image and engine artifacts valid |
| Move message/request objects instead of copying | Stalls | Request copying not causal |
| Replace in-memory loader with temporary-file loader | Stalls | Decoder path and JPEG lifetime not causal |
| Use native short prompt | Stalls | Prompt length not causal |
| Reduce generation limit to 64 | Stalls | Failure occurs before generation |
| CUDA-GDB during failure | Fused-attention kernel remains active | Stall is inside TensorRT prefill |
| Standalone app consuming shared `libcosmos_trt_backend.so` | Stalls | ROS node, executor, and rosbag traffic not required |
| Compile backend directly into a ROS-free executable | Succeeds, ~1.45 s | Final executable device linking matters |
| Run direct-linked backend on `std::thread` | Succeeds, ~1.45 s | Worker-thread execution is safe |
| Link successful direct executable to `rclcpp` without calling `rclcpp::init()` | Stalls | Loading ROS's native dependency set is sufficient |
| Put ROS and Edge-LLM in separate processes | Succeeds repeatedly | Process co-loading is the supported failure boundary |
| Restore `loadImageFromMemory()` in isolated worker | Succeeds repeatedly | Temporary-file workaround unnecessary |

## OpenCV ABI hazard

The original ROS process loaded ROS `cv_bridge` with OpenCV 4.6 and the
JetPack stack with NVIDIA OpenCV 4.8. Passing C++ OpenCV objects across mixed
ABI versions is unsafe even though this did not uniquely explain the GPU
stall. The production node removed `cv_bridge`, directly converts supported raw
encodings, and no longer installs `ros-jazzy-cv-bridge`.

This also avoids APT transactions that downgrade NVIDIA OpenCV and remove
JetPack development packages.

## Benign action-engine message

Successful and failing runs both logged:

```text
Failed to load Action runner .../engine/action/action.engine:
No such file or directory
```

The action runner is optional. The validated Cosmos-Reason2-8B bundle has no
action head, and image reasoning completes without it.

## Production fix

Production launch starts `edge_vlm_ros_node` and `edge_vlm_server`.

The ROS process:

- links ROS and OpenCV, but not CUDA, TensorRT, or Edge-LLM;
- maintains a newest-frame queue of depth one;
- converts raw ROS images to packed BGR8;
- sends bounded requests over a Unix socket;
- publishes `VlmResult`.

The GPU worker:

- links Edge-LLM, CuTe DSL, TensorRT, CUDA, and NVIDIA OpenCV;
- does not link ROS, RMW, DDS, or `cv_bridge`;
- contains `sm_110a` CUDA images;
- keeps engines and the CUDA context persistent;
- performs in-memory JPEG loading and serial inference.

Deployment verification checks both dynamic dependency boundaries.

## End-to-end validation

The isolated worker completed repeated NVIDIA image-proc rosbag requests with
the default long prompt and 256-token output limit:

```text
frame 1: 3.259 seconds
frame 3: 5.913 seconds
frame 5: 3.715 seconds
```

A later run using the final in-memory loading path completed:

```text
frame 1: 3.202 seconds
frame 4: 3.915 seconds
frame 5: 3.179 seconds
```

Both TensorRT optimization-profile transitions completed on every request, and
the ROS process published coherent warehouse descriptions. Latency varied with
output length; these observations are not a formal benchmark.

## Crash-recovery validation

IPC writes use `send(..., MSG_NOSIGNAL)`. The automated recovery test:

1. obtained a successful result;
2. killed `edge_vlm_server` with `SIGKILL`;
3. observed launch create a worker with a new PID;
4. allowed the current frame to fail without replay;
5. reconnected on a subsequent frame;
6. received successful results without restarting `edge_vlm_ros_node`.

Post-recovery requests with a 64-token limit completed in approximately:

```text
1.573 s
1.537 s
1.547 s
1.576 s
```

This validates worker-exit recovery, socket-peer loss handling, respawn, and
reconnection.

## Residual risk and follow-up

A 90-second socket deadline prevents the ROS inference thread from waiting
forever. It does not terminate a worker that remains alive while blocked inside
a GPU call. Since launch only respawns an exited process, active watchdog
termination remains future work in issue #6.

Other follow-ups:

- IPC protocol and fake-worker tests: issue #13;
- repeatable performance benchmarks: issue #7;
- temporal video reasoning: issue #8;
- model portability and optimization: issue #9;
- RViz2 visualization: issue #10;
- task-level quality evaluation: issue #11;
- configurable prompts and bounded context: issue #12.

## Final conclusion

The deployed rosbag-to-Cosmos pipeline is functional on the validated Thor
stack. The evidence supports this precise conclusion:

> TensorRT Edge-LLM and the ROS 2 native dependency set must remain in separate
> processes on this validated platform.

The evidence does not identify one exact ROS shared object, symbol, or TensorRT
implementation defect. The isolated, direct-linked worker is the tested
production solution and the correct basis for future work.
