# Jetson AGX Thor Edge-LLM prefill stall RCA

Status: isolated-worker production path validated on Thor  
Platform: NVIDIA Jetson AGX Thor  
Date opened: 2026-07-30

## Summary

The ROS 2 Cosmos reasoner successfully initializes the TensorRT Edge-LLM runtime, loads the Cosmos-Reason2-8B LLM and visual engines, receives and preprocesses a camera frame, and then stalls indefinitely during base-model prefill.

The same engines and the same extracted camera image complete normally with NVIDIA's native `llm_inference` executable. A standalone smoke test that links against this repository's installed `libcosmos_trt_backend.so`, without initializing ROS, reproduces the stall. A direct-linked executable that compiles the same backend implementation into the final executable completes successfully in approximately 1.45 seconds. This confirms that the failing boundary is the intermediate `libcosmos_trt_backend.so` CUDA/TensorRT linkage, rather than ROS, the rosbag, the request implementation, or the model artifacts.

## System configuration

| Component | Observed value |
| --- | --- |
| Device | NVIDIA Jetson AGX Thor |
| Architecture | aarch64 |
| GPU compute capability | 11.0 |
| Jetson Linux | R39.2 |
| JetPack | 7.2 |
| CUDA | 13.2 |
| Driver | 595.78 |
| ROS 2 | Jazzy |
| Isaac ROS | 4.5 packages, with NVIDIA compatibility warning on JetPack 7.2 |
| Model | Cosmos-Reason2-8B |
| Quantization | NVFP4 |
| LLM engine maximum input | 1024 tokens |
| LLM engine KV capacity | 4096 tokens |

Resolved deployment paths during testing:

```text
Edge-LLM source:   /home/daniel/TensorRT-Edge-LLM
Edge-LLM build:    /home/daniel/TensorRT-Edge-LLM/build
Core archive:      /home/daniel/TensorRT-Edge-LLM/build/cpp/libedgellmCore.a
Plugin:            /home/daniel/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
Model engine:      /home/daniel/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engine
LLM engine:        /home/daniel/tensorrt-edgellm-workspace/Cosmos-Reason2-8B/engine/llm
ROS workspace:     /home/daniel/ros2_ws
Installed backend: /home/daniel/ros2_ws/install/cosmos_ros2_video_reasoner/lib/libcosmos_trt_backend.so
```

## Initial CUDA architecture failure

The first ROS runs failed during runtime construction with:

```text
CUDA runtime error in cudaOccupancyMaxActiveBlocksPerMultiprocessor(...): device kernel image is invalid
```

Inspection showed stale or incorrectly device-linked CUDA code:

```text
NVIDIA Thor compute capability: 11.0
Edge-LLM RoPE object:            sm_110a
Initial ROS backend cubin:       sm_75
```

The Edge-LLM build and ROS backend were rebuilt for `sm_110a`. The installed backend now reports:

```text
libcosmos_trt_backend.1.sm_110a.cubin
```

This fixed runtime construction. The runtime now allocates tensors, loads the tokenizer and visual engine, captures decoding graphs, and subscribes to the ROS image topic.

## Current failure signature

After receiving the first image, output stops after messages resembling:

```text
Processing vision inputs
Switching optimization profile from: 1 to 0
```

GPU compute utilization remains near 98 percent until the process is killed. On Thor, `nvidia-smi` reports GPU compute utilization but does not provide normal discrete-GPU memory accounting. A short utilization spike is expected; sustained utilization for more than a minute is not.

CUDA-GDB shows the host inference thread blocked at:

```text
cudaStreamSynchronize
trt_edgellm::rt::LLMInferenceRuntime::runBaseModelPrefill
trt_edgellm::rt::LLMInferenceRuntime::handleRequest
cosmos_ros2_video_reasoner::TensorRTEdgeLLMBackend::infer
```

The active GPU kernel is a TensorRT Blackwell fused-attention kernel:

```text
kernel_cutlass_kernel___main__BlackwellFusedMultiHeadAttentionForward...
```

Observed launch geometry:

```text
grid:  (22, 1, 1)
block: (512, 1, 1)
status: Active indefinitely
```

No contemporaneous NVRM Xid, MMU fault, or GPU channel exception was found. Older NVRM messages in `dmesg` were emitted during system boot and were not correlated with the test.

## Camera input validation

The NVIDIA image-proc rosbag publishes:

```text
height:       1200
width:        1920
encoding:     rgb8
step:         5760
payload size: 6912000 bytes
```

`step` is exactly `width * 3`, so the source image has no unexpected row padding. The frame was captured through `cv_bridge`, written to JPEG, and validated as a 1920x1200 three-component JPEG.

The wrapper's image-size limit was reduced from 1280 pixels to 512 pixels, producing a 512x320 inference image. The stall remained.

## Successful native control test

The exact captured Hawk frame was resized to 512x320 and used with NVIDIA's native executable:

```bash
./build/examples/llm/llm_inference \
  --engineDir "$WORKSPACE_DIR/$MODEL_NAME/engine/llm" \
  --multimodalEngineDir "$WORKSPACE_DIR/$MODEL_NAME/engine" \
  --inputFile /tmp/input_hawk.json \
  --outputFile /tmp/output_hawk.json \
  --maxGenerateLength 64 \
  --dumpOutput
```

The native executable completed in approximately 1.6 seconds and returned a coherent warehouse-scene description. Its visual profile changed from `1` to `0` and back from `0` to `1` before generation.

This control establishes that the following artifacts are functional together:

- Cosmos-Reason2-8B LLM engine
- Cosmos visual engine
- Edge-LLM plugin
- rebuilt `sm_110a` Edge-LLM code
- captured Hawk image
- 512x320 image shape
- tokenizer and chat template
- CUDA 13.2 runtime and driver

## Experiments and results

| Experiment | Result | Conclusion |
| --- | --- | --- |
| Rebuild Edge-LLM and ROS backend for `sm_110a` | Runtime construction succeeds | Initial architecture mismatch fixed |
| Reduce image width from 1280 to 512 | Still stalls | Input resolution is not the primary cause |
| Capture exact rosbag frame and run native `llm_inference` | Succeeds | Image and engine artifacts are valid |
| Move messages and inner request rather than copying them | Still stalls | Request copy semantics were not the cause |
| Replace `loadImageFromMemory()` with temporary-file `loadImageFromFile()` | Still stalls | Memory image decoder and JPEG-buffer lifetime are not the cause |
| Match native short prompt | Still stalls | Default prompt length is not the cause |
| Reduce maximum generation length to 64 | Still stalls | Generation length is not the cause |
| CUDA-GDB attach during stall | Fused attention remains active; host waits in prefill | Failure is inside TensorRT base-model prefill |
| Run standalone program linked to installed `libcosmos_trt_backend.so` without ROS initialization | Still stalls | ROS, DDS, rosbag playback, executor, and cv_bridge are not required to reproduce |
| Run NVIDIA native `llm_inference` with same engine and image | Succeeds | Failure follows this repository's embedded/shared backend path |
| Compile the same backend implementation directly into a final executable | Succeeds in approximately 1.45 seconds | Intermediate shared-library CUDA device linking is a confirmed failure boundary |
| Inspect production executable dependencies | OpenCV 4.6 and 4.8 loaded together through cv_bridge and NVIDIA OpenCV | A second C++ ABI hazard exists in the ROS image-conversion path |
| Run direct-linked backend entirely on a `std::thread` without ROS libraries | Succeeds in approximately 1.45 seconds | Worker-thread execution is safe |
| Link the successful direct smoke executable to `rclcpp` but do not call `rclcpp::init()` | Stalls in the same fused-attention prefill kernel | Loading the ROS dependency set is sufficient to trigger the failure |

## Benign action-runner message

Runtime initialization logs:

```text
Failed to load Action runner .../engine/action/action.engine: No such file or directory
```

This is an optional engine probe. Cosmos-Reason2-8B was exported without an action head, and initialization intentionally continues. The same message appears in successful native inference, so it is unrelated to the stall.

## Current localization

The best-supported boundary is now:

```text
NVIDIA native llm_inference executable
  -> succeeds

Repository libcosmos_trt_backend.so
  -> TensorRT fused-attention prefill kernel remains active indefinitely

Standalone consumer of libcosmos_trt_backend.so
  -> reproduces without ROS
```

The leading hypothesis is a CUDA/TensorRT linkage or device-registration difference caused by embedding the static Edge-LLM core and CuTe DSL artifacts in a shared library. This is not yet proven. ABI differences, link order, symbol interposition, and executable-versus-shared-object CUDA device linking remain candidates.

## Next RCA steps

1. Build the production `cosmos_reasoner` executable with `tensorrt_edge_llm_backend.cpp` compiled directly into it.
2. Link Edge-LLM core, CuTe DSL, TensorRT, and CUDA directly to `cosmos_reasoner`.
3. Resolve CUDA device symbols for `sm_110a` at the final executable boundary.
4. Verify that `ldd cosmos_reasoner` no longer resolves `libcosmos_trt_backend.so`.
5. Verify the final executable contains an `sm_110a` cubin.
6. Repeat the image-proc rosbag end-to-end test and confirm publication on `/cosmos/reasoning`.
7. After validation, remove the obsolete shared backend or retain it only behind an explicit diagnostic build option.

## Temporary diagnostic changes

Some local Thor tests changed the backend to:

- move message/request objects explicitly;
- write `/tmp/cosmos_ros2_frame.jpg`;
- use `loadImageFromFile()` instead of `loadImageFromMemory()`;
- use a 512-pixel image-width limit;
- use a 64-token generation limit.

These changes were diagnostic and should not all be treated as the final production solution. In particular, writing every frame to `/tmp` is not acceptable for deployment. Revert or replace temporary diagnostics after the linkage root cause is confirmed.


## OpenCV ABI conflict

Linker and `ldd` inspection found that the ROS executable loaded two OpenCV C++ runtimes:

```text
libcv_bridge.so
libopencv_core.so.406
libopencv_imgproc.so.406
libopencv_imgcodecs.so.406
libopencv_core.so.408
libopencv_imgproc.so.408
libopencv_imgcodecs.so.408
```

ROS Jazzy's binary `cv_bridge` package is built against Ubuntu OpenCV 4.6, while the Thor JetPack stack provides NVIDIA OpenCV 4.8. Passing `cv::Mat` objects across this mixed C++ ABI boundary is unsafe and may cause memory corruption or undefined behavior.

The production node therefore removes its `cv_bridge` runtime dependency and converts supported `sensor_msgs/msg/Image` encodings directly with the selected NVIDIA OpenCV runtime. Supported encodings are `bgr8`, `rgb8`, and `mono8`; row stride and payload size are validated before constructing an OpenCV view. This avoids downgrading NVIDIA OpenCV or removing JetPack packages.


## ROS library-load isolation result

A direct-linked smoke executable was validated in both main-thread and `std::thread` modes; both completed in approximately 1.45 seconds. The same executable was then linked to `rclcpp` without calling `rclcpp::init()` or constructing any ROS node. It stalled in the same TensorRT Blackwell fused-attention prefill kernel.

This establishes that ROS executor behavior, DDS traffic, node creation, callback scheduling, and CUDA thread affinity are not required to reproduce the failure. Loading the transitive ROS 2 C++ dependency set into the Edge-LLM process is sufficient.

The production architecture must therefore isolate Edge-LLM from ROS in a separate process unless the exact conflicting shared object or symbol-interposition issue is identified and proven safe. The preferred deployment is a persistent ROS-free inference worker, direct-linked to Edge-LLM at its final executable boundary, with the ROS node using a bounded IPC protocol. Process isolation also allows a wedged GPU worker to be terminated and restarted without wedging ROS shutdown.


## Isolated C++ production architecture

The draft fix now separates the runtime into two native processes:

```text
cosmos_reasoner (ROS 2 / rclcpp)
  -> versioned Unix-domain socket protocol
cosmos_inference_worker (ROS-free, direct-linked Edge-LLM)
```

The ROS process no longer links TensorRT, CUDA, Edge-LLM, CuTe DSL, or the Edge-LLM plugin. The worker does not link ROS, RMW, or DDS libraries. Engines and the CUDA context remain persistent in the worker.

IPC requests carry a monotonically increasing request ID, packed BGR8 image bytes, dimensions, prompt, and generation parameters. Responses carry the matching request ID, success state, text or error, and inference duration. The ROS node retains its bounded newest-frame queue and publishes the existing `VisionReasoningResult` message.

This process boundary is intended both to avoid the confirmed library-load interaction and to permit future worker timeout/restart supervision without wedging ROS shutdown.


## Successful end-to-end isolated-worker validation

On 2026-07-30, the isolated C++ worker architecture completed the NVIDIA image-proc rosbag test on Thor. The launch started two processes:

```text
cosmos_inference_worker
cosmos_reasoner
```

The worker initialized Cosmos-Reason2-8B once, accepted multiple IPC requests, completed both TensorRT optimization-profile transitions for every frame, and returned coherent warehouse-scene descriptions. The ROS process logged and published successful results.

Observed inference durations with the long default prompt and 256-token generation limit were approximately:

```text
frame 1: 3.259 seconds
frame 3: 5.913 seconds
frame 5: 3.715 seconds
```

The variation primarily reflects output length; the IPC transport is not implicated. Multiple sequential successful requests also confirm that the worker keeps its engines and CUDA context persistent.

This validates the core fix: keep ROS and its transitive native libraries out of the Edge-LLM process. The isolated worker was then restored to the deployment-safe in-memory image path, using `loadImageFromMemory()` and move-only request construction. The same rosbag completed three sequential requests successfully:

```text
frame 1: 3.202 seconds
frame 4: 3.915 seconds
frame 5: 3.179 seconds
```

This eliminates the temporary `/tmp/cosmos_ros2_frame.jpg` workaround and confirms that in-memory decoding is reliable once ROS libraries and Edge-LLM are separated by the process boundary.

## IPC timeout and restart policy

The ROS-side IPC client now applies bounded send and receive timeouts to every worker connection. The default request deadline is 90 seconds, compared with observed inference times below 6 seconds in the validation runs. If a transport operation times out or fails, the client closes the connection and reports the current frame as failed. It does not automatically replay that frame because the worker may have partially executed it; the next sampled frame establishes a new connection.

The launch description now respawns `cosmos_inference_worker` two seconds after an unexpected worker exit. This covers worker crashes and externally terminated wedged workers while preserving the ROS process. A future supervisor can add automatic termination of a still-alive GPU worker after a request deadline; the current deadline prevents the ROS inference thread from waiting indefinitely but does not itself kill a process stuck inside an uninterruptible GPU call.

Remaining production hardening includes clean shutdown and socket cleanup tests, IPC protocol tests, an active worker watchdog for live-but-wedged processes, and repeatable latency/throughput benchmarks.


## Worker crash-recovery validation

The automated recovery test was run on Thor after replacing IPC writes with `send(..., MSG_NOSIGNAL)`. The test obtained a successful result, terminated `cosmos_inference_worker` with `SIGKILL`, observed launch create a worker with a new PID, and then received another successful result without restarting `cosmos_reasoner`.

Post-recovery inference continued across multiple frames:

```text
frame 5: 1.573 seconds
frame 7: 1.537 seconds
frame 8: 1.547 seconds
frame 10: 1.576 seconds
```

These measurements used a 64-token generation limit. The test confirms:

- worker process respawn works;
- the ROS process survives loss of the Unix-socket peer;
- `SIGPIPE` is suppressed;
- the IPC client reconnects on a subsequent frame;
- the replacement worker reinitializes the engines and serves repeated requests.

This validates recovery from worker exit. It does not yet validate automatic recovery from a live worker wedged inside a GPU call. That case requires an external watchdog capable of terminating the worker after the ROS-side request deadline.
