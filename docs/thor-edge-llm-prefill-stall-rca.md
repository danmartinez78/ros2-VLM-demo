# Jetson AGX Thor Edge-LLM prefill stall RCA

Status: root cause localized; production fix under validation  
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
| Run NVIDIA native `llm_inference` with same engine and image | Succeeds | Failure follows this repository's embedded/shared backend path |\n| Compile the same backend implementation directly into a final executable | Succeeds in approximately 1.45 seconds | Intermediate shared-library CUDA device linking is a confirmed failure boundary |\n| Inspect production executable dependencies | OpenCV 4.6 and 4.8 loaded together through cv_bridge and NVIDIA OpenCV | A second C++ ABI hazard exists in the ROS image-conversion path |

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
