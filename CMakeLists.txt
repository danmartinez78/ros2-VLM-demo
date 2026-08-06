cmake_minimum_required(VERSION 3.20)
project(edge_vlm_ros LANGUAGES C CXX)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── ament ─────────────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(yolo_msgs QUIET)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(pluginlib QUIET)
find_package(rviz_common QUIET)
find_package(Qt5 QUIET COMPONENTS Core Gui Widgets)

# ── Custom message ────────────────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedObject.msg"
  "msg/TrackedObservation.msg"
  "msg/VlmResult.msg"
  DEPENDENCIES std_msgs builtin_interfaces sensor_msgs
)

# ── TensorRT Edge-LLM integration ─────────────────────────────────────────────
#
# Set these cache variables when invoking CMake:
#
#   -DTENSORRT_EDGE_LLM_ROOT=$HOME/TensorRT-Edge-LLM
#   -DTENSORRT_EDGE_LLM_BUILD_DIR=$HOME/TensorRT-Edge-LLM/build
#
# All required headers and libraries are validated at configure time.
# If TensorRT_EDGE_LLM_ROOT is not set, the TensorRT backend target is skipped
# and only hardware-independent targets (node library + tests) are built.

set(TENSORRT_EDGE_LLM_ROOT "" CACHE PATH
  "Root directory of the checked-out TensorRT-Edge-LLM repository")
set(TENSORRT_EDGE_LLM_BUILD_DIR "" CACHE PATH
  "Build directory of TensorRT-Edge-LLM (contains libedgellmCore.a, etc.)")
set(TENSORRT_EDGE_LLM_CUDA_ARCHITECTURES "110a" CACHE STRING
  "CUDA architectures used for the Edge-LLM device-link step (Thor: 110a)")

set(BUILD_TENSORRT_BACKEND FALSE)

if(TENSORRT_EDGE_LLM_ROOT)
  message(STATUS "TensorRT Edge-LLM root : ${TENSORRT_EDGE_LLM_ROOT}")

  # ── Locate TRT package dir (contains TRT headers and libs) ───────────────
  if(NOT TRT_PACKAGE_DIR)
    message(FATAL_ERROR
      "Please set -DTRT_PACKAGE_DIR to the TensorRT package directory "
      "(containing include/NvInfer.h and lib/libnvinfer.so).")
  endif()

  # ── Validate required Edge-LLM headers ──────────────────────────────────
  set(_EDGELLM_REQUIRED_HEADERS
    "cpp/common/trtUtils.h"
    "cpp/runtime/llmInferenceRuntime.h"
    "cpp/runtime/llmRuntimeUtils.h"
    "cpp/runtime/imageUtils.h"
    "cpp/tokenizer/tokenizer.h"
    "cpp/multimodal/multimodalRunner.h"
    "3rdParty/stb/stb_image.h"
  )
  foreach(_hdr IN LISTS _EDGELLM_REQUIRED_HEADERS)
    if(NOT EXISTS "${TENSORRT_EDGE_LLM_ROOT}/${_hdr}")
      message(FATAL_ERROR
        "Required TensorRT-Edge-LLM header not found: "
        "${TENSORRT_EDGE_LLM_ROOT}/${_hdr}\n"
        "Set TENSORRT_EDGE_LLM_ROOT to the repository root.")
    endif()
  endforeach()

  # ── Validate required TRT header ─────────────────────────────────────────
  find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS ${TRT_PACKAGE_DIR}
    PATH_SUFFIXES include ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
    REQUIRED)

  # ── Validate required Edge-LLM built libraries ───────────────────────────
  if(TENSORRT_EDGE_LLM_BUILD_DIR)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # TensorRT Edge-LLM 0.9.x builds tokenizer sources into edgellmCore.
    # A separate tokenizer archive is no longer produced.
    find_library(EDGELLM_CORE_LIB edgellmCore
      HINTS "${TENSORRT_EDGE_LLM_BUILD_DIR}"
      PATH_SUFFIXES "" cpp lib
      REQUIRED)

    # Edge-LLM's own CMake target carries the CuTe DSL AOT kernel archive as
    # a transitive dependency. Consumers of the raw libedgellmCore.a archive
    # must restore that dependency explicitly or the final executable link
    # fails with unresolved _mlir_gemm_* symbols.
    set(CUTE_DSL_ARTIFACT_TAG "" CACHE STRING
      "CuTe DSL artifact tag (for Thor this is normally sm_110)")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      set(_CUTE_DSL_ARCH "aarch64")
    else()
      set(_CUTE_DSL_ARCH "x86_64")
    endif()

    set(_CUTE_DSL_ARTIFACT_ROOT
      "${TENSORRT_EDGE_LLM_ROOT}/cpp/kernels/cuteDSLArtifact/${_CUTE_DSL_ARCH}")
    set(_CUTE_DSL_ARTIFACT_DIR "")

    if(CUTE_DSL_ARTIFACT_TAG)
      set(_CUTE_DSL_ARTIFACT_DIR
        "${_CUTE_DSL_ARTIFACT_ROOT}/${CUTE_DSL_ARTIFACT_TAG}")
    elseif(EXISTS "${_CUTE_DSL_ARTIFACT_ROOT}/metadata.json")
      set(_CUTE_DSL_ARTIFACT_DIR "${_CUTE_DSL_ARTIFACT_ROOT}")
    else()
      file(GLOB _CUTE_DSL_CANDIDATES LIST_DIRECTORIES TRUE
        "${_CUTE_DSL_ARTIFACT_ROOT}/*")
      set(_CUTE_DSL_MATCHES "")
      foreach(_candidate IN LISTS _CUTE_DSL_CANDIDATES)
        if(IS_DIRECTORY "${_candidate}"
           AND EXISTS "${_candidate}/metadata.json")
          list(APPEND _CUTE_DSL_MATCHES "${_candidate}")
        endif()
      endforeach()
      list(LENGTH _CUTE_DSL_MATCHES _CUTE_DSL_MATCH_COUNT)
      if(_CUTE_DSL_MATCH_COUNT EQUAL 1)
        list(GET _CUTE_DSL_MATCHES 0 _CUTE_DSL_ARTIFACT_DIR)
      elseif(_CUTE_DSL_MATCH_COUNT GREATER 1)
        message(FATAL_ERROR
          "Multiple CuTe DSL artifact directories found under "
          "${_CUTE_DSL_ARTIFACT_ROOT}: ${_CUTE_DSL_MATCHES}\n"
          "Set -DCUTE_DSL_ARTIFACT_TAG=sm_110 for Jetson AGX Thor.")
      endif()
    endif()

    set(CUTE_DSL_STATIC_LIB
      "${_CUTE_DSL_ARTIFACT_DIR}/libcutedsl_${_CUTE_DSL_ARCH}.a")
    set(CUTE_DSL_RUNTIME_LIB
      "${_CUTE_DSL_ARTIFACT_DIR}/libcute_dsl_runtime.so")

    if(NOT _CUTE_DSL_ARTIFACT_DIR OR NOT EXISTS "${CUTE_DSL_STATIC_LIB}")
      message(FATAL_ERROR
        "TensorRT Edge-LLM was built with CuTe DSL, but its AOT kernel archive "
        "was not found under ${_CUTE_DSL_ARTIFACT_ROOT}.\n"
        "Rebuild Edge-LLM with -DENABLE_CUTE_DSL=ALL, or set "
        "-DCUTE_DSL_ARTIFACT_TAG=sm_110 for Jetson AGX Thor.")
    endif()

    message(STATUS "CuTe DSL artifact   : ${_CUTE_DSL_ARTIFACT_DIR}")

    find_library(NVINFER_LIB nvinfer
      HINTS ${TRT_PACKAGE_DIR}
      PATH_SUFFIXES lib lib64 ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
      REQUIRED)

    find_library(NVONNXPARSER_LIB nvonnxparser
      HINTS ${TRT_PACKAGE_DIR}
      PATH_SUFFIXES lib lib64 ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
      REQUIRED)

    # nlohmann/json bundled with Edge-LLM
    set(_NLOHMANN_JSON_INCLUDE "${TENSORRT_EDGE_LLM_ROOT}/3rdParty/nlohmannJson/include")
    if(NOT EXISTS "${_NLOHMANN_JSON_INCLUDE}/nlohmann/json.hpp")
      message(FATAL_ERROR
        "nlohmann/json header not found at ${_NLOHMANN_JSON_INCLUDE}. "
        "Run 'git submodule update --init' inside TensorRT-Edge-LLM.")
    endif()

    set(BUILD_TENSORRT_BACKEND TRUE)
  else()
    message(WARNING
      "TENSORRT_EDGE_LLM_BUILD_DIR not set. "
      "TensorRT backend will NOT be built. "
      "Set -DTENSORRT_EDGE_LLM_BUILD_DIR=<path> to enable it.")
  endif()
endif()

# ── Node library (hardware-independent) ──────────────────────────────────────
add_library(vlm_reasoner_node SHARED
  src/vlm_reasoner_node.cpp
)

target_include_directories(vlm_reasoner_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(vlm_reasoner_node
  rclcpp
  rcl_interfaces
  sensor_msgs
  std_msgs
  builtin_interfaces
  image_transport
)

target_link_libraries(vlm_reasoner_node
  ${OpenCV_LIBRARIES}
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(vlm_reasoner_node "${cpp_typesupport_target}")

add_library(edge_vlm_rviz_formatting SHARED
  src/vision_reasoning_rviz_formatting.cpp
)
target_include_directories(edge_vlm_rviz_formatting PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(edge_vlm_rviz_formatting
  rclcpp
  std_msgs
  builtin_interfaces
)
target_link_libraries(edge_vlm_rviz_formatting "${cpp_typesupport_target}")

set(BUILD_RVIZ2_PANEL FALSE)
if(pluginlib_FOUND AND rviz_common_FOUND AND Qt5_FOUND)
  set(BUILD_RVIZ2_PANEL TRUE)
endif()

if(BUILD_RVIZ2_PANEL)
  add_library(edge_vlm_rviz_panel SHARED
    src/vision_reasoning_rviz_panel.cpp
  )
  set_target_properties(edge_vlm_rviz_panel PROPERTIES AUTOMOC ON)
  target_include_directories(edge_vlm_rviz_panel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(edge_vlm_rviz_panel
    edge_vlm_rviz_formatting
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
  )
  ament_target_dependencies(edge_vlm_rviz_panel
    pluginlib
    rclcpp
    rviz_common
    sensor_msgs
    std_msgs
    builtin_interfaces
  )
  target_link_libraries(edge_vlm_rviz_panel "${cpp_typesupport_target}")
  pluginlib_export_plugin_description_file(
    rviz_common
    rviz/vision_reasoning_panel_plugins.xml)
else()
  message(STATUS
    "RViz2 panel disabled. Install rviz_common + Qt5 + pluginlib to build "
    "edge_vlm_rviz_panel.")
endif()

# ── IPC clients (hardware-independent) ──────────────────────────────────────
# These targets never link TensorRT, CUDA, or Edge-LLM. Keeping them outside the
# hardware condition lets CI compile the service contract and client tools.
add_library(edge_vlm_ipc_client STATIC src/ipc_inference_backend.cpp)
target_include_directories(edge_vlm_ipc_client PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(edge_vlm_ipc_client ${OpenCV_LIBRARIES})

add_executable(edge_vlm_ros_node src/main.cpp)
target_include_directories(edge_vlm_ros_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(edge_vlm_ros_node
  edge_vlm_ipc_client
  vlm_reasoner_node)
ament_target_dependencies(edge_vlm_ros_node rclcpp)

add_library(tracked_observation_adapter SHARED
  src/tracked_observation_adapter.cpp
)
target_include_directories(tracked_observation_adapter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(tracked_observation_adapter
  message_filters
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
)
target_link_libraries(tracked_observation_adapter "${cpp_typesupport_target}")

add_executable(edge_vlm_tracked_observation_adapter src/tracked_observation_adapter_main.cpp)
target_include_directories(edge_vlm_tracked_observation_adapter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(edge_vlm_tracked_observation_adapter
  tracked_observation_adapter)
ament_target_dependencies(edge_vlm_tracked_observation_adapter rclcpp)

set(BUILD_YOLO_ADAPTER FALSE)
if(yolo_msgs_FOUND)
  set(BUILD_YOLO_ADAPTER TRUE)
  add_library(yolo_detection2d_adapter SHARED
    src/yolo_detection2d_adapter.cpp
  )
  target_include_directories(yolo_detection2d_adapter PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(yolo_detection2d_adapter
    rcl_interfaces
    rclcpp
    vision_msgs
    yolo_msgs
  )

  add_executable(edge_vlm_yolo_detection2d_adapter src/yolo_detection2d_adapter_main.cpp)
  target_include_directories(edge_vlm_yolo_detection2d_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(edge_vlm_yolo_detection2d_adapter
    yolo_detection2d_adapter)
  ament_target_dependencies(edge_vlm_yolo_detection2d_adapter rclcpp)
else()
  message(STATUS "yolo_msgs not found. YOLO Detection2D adapter will not be built.")
endif()

add_executable(edge_vlm_cli src/reasoning_cli.cpp)
target_include_directories(edge_vlm_cli PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(edge_vlm_cli
  edge_vlm_ipc_client
  ${OpenCV_LIBRARIES})

# ── TensorRT backend library (hardware-dependent) ────────────────────────────
if(BUILD_TENSORRT_BACKEND)
  add_library(edge_vlm_trt_backend SHARED
    src/tensorrt_edge_llm_backend.cpp
  )

  target_include_directories(edge_vlm_trt_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TENSORRT_EDGE_LLM_ROOT}/cpp
    ${TENSORRT_EDGE_LLM_ROOT}/examples
    ${TRT_INCLUDE_DIR}
    ${_NLOHMANN_JSON_INCLUDE}
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/stb
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/miniaudio
    ${TENSORRT_EDGE_LLM_ROOT}/examples/multimodal
  )

  target_link_libraries(edge_vlm_trt_backend
    vlm_reasoner_node
    ${EDGELLM_CORE_LIB}
    ${CUTE_DSL_STATIC_LIB}
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    CUDA::cudart
    CUDA::cuda_driver
    dl
    ${OpenCV_LIBRARIES}
  )

  ament_target_dependencies(edge_vlm_trt_backend rclcpp rcl_interfaces)
  if(EXISTS "${CUTE_DSL_RUNTIME_LIB}")
    target_link_libraries(edge_vlm_trt_backend
      ${CUTE_DSL_RUNTIME_LIB})
  endif()

  set_target_properties(edge_vlm_trt_backend PROPERTIES
    CUDA_ARCHITECTURES "${TENSORRT_EDGE_LLM_CUDA_ARCHITECTURES}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INSTALL_RPATH_USE_LINK_PATH TRUE)

  # ── Direct-linked diagnostic executable ───────────────────────────────
  # Compiles the backend implementation into the final executable instead of
  # consuming libedge_vlm_trt_backend.so. This isolates CUDA device-link and
  # shared-library behavior from the backend implementation itself.
  add_executable(edge_vlm_backend_direct_smoke
    src/backend_direct_smoke.cpp
    src/tensorrt_edge_llm_backend.cpp
  )
  target_include_directories(edge_vlm_backend_direct_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_EDGE_LLM_ROOT}/cpp
    ${TENSORRT_EDGE_LLM_ROOT}/examples
    ${TRT_INCLUDE_DIR}
    ${_NLOHMANN_JSON_INCLUDE}
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/stb
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/miniaudio
    ${TENSORRT_EDGE_LLM_ROOT}/examples/multimodal
  )
  ament_target_dependencies(edge_vlm_backend_direct_smoke rclcpp)
  target_link_libraries(edge_vlm_backend_direct_smoke
    ${EDGELLM_CORE_LIB}
    ${CUTE_DSL_STATIC_LIB}
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    CUDA::cudart
    CUDA::cuda_driver
    dl
    ${OpenCV_LIBRARIES}
  )
  if(EXISTS "${CUTE_DSL_RUNTIME_LIB}")
    target_link_libraries(edge_vlm_backend_direct_smoke
      ${CUTE_DSL_RUNTIME_LIB})
  endif()
  set_target_properties(edge_vlm_backend_direct_smoke PROPERTIES
    CUDA_ARCHITECTURES "${TENSORRT_EDGE_LLM_CUDA_ARCHITECTURES}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INSTALL_RPATH_USE_LINK_PATH TRUE)

  # ── ROS-free persistent inference worker ──────────────────────────────
  add_executable(edge_vlm_server
    src/inference_worker.cpp
    src/tensorrt_edge_llm_backend.cpp
  )
  target_include_directories(edge_vlm_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_EDGE_LLM_ROOT}/cpp
    ${TENSORRT_EDGE_LLM_ROOT}/examples
    ${TRT_INCLUDE_DIR}
    ${_NLOHMANN_JSON_INCLUDE}
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/stb
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/miniaudio
    ${TENSORRT_EDGE_LLM_ROOT}/examples/multimodal
  )
  target_link_libraries(edge_vlm_server
    ${EDGELLM_CORE_LIB}
    ${CUTE_DSL_STATIC_LIB}
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    CUDA::cudart
    CUDA::cuda_driver
    dl
    ${OpenCV_LIBRARIES}
  )
  if(EXISTS "${CUTE_DSL_RUNTIME_LIB}")
    target_link_libraries(edge_vlm_server ${CUTE_DSL_RUNTIME_LIB})
  endif()
  set_target_properties(edge_vlm_server PROPERTIES
    CUDA_ARCHITECTURES "${TENSORRT_EDGE_LLM_CUDA_ARCHITECTURES}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INSTALL_RPATH_USE_LINK_PATH TRUE)

  install(TARGETS edge_vlm_trt_backend edge_vlm_backend_direct_smoke edge_vlm_server
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )

  # Set RPATH so the plugin library is found at runtime without LD_LIBRARY_PATH
  set_target_properties(edge_vlm_ros_node PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
  )
else()
  message(STATUS
    "TensorRT backend disabled. "
    "Building node library and tests only (no GPU required).")
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS edge_vlm_ros_node edge_vlm_tracked_observation_adapter edge_vlm_cli edge_vlm_ipc_client
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_YOLO_ADAPTER)
  install(TARGETS edge_vlm_yolo_detection2d_adapter yolo_detection2d_adapter
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(TARGETS vlm_reasoner_node
  EXPORT vlm_reasoner_nodeTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS tracked_observation_adapter
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS edge_vlm_rviz_formatting
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_RVIZ2_PANEL)
  install(TARGETS edge_vlm_rviz_panel
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  # Linting
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Frame sampling tests (no GPU)
  ament_add_gtest(test_frame_sampling
    test/test_frame_sampling.cpp
  )
  target_include_directories(test_frame_sampling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )
  target_link_libraries(test_frame_sampling vlm_reasoner_node)
  ament_target_dependencies(test_frame_sampling rclcpp sensor_msgs)

  # Node integration tests (no GPU)
  ament_add_gtest(test_node
    test/test_node.cpp
  )
  target_include_directories(test_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )
  target_link_libraries(test_node vlm_reasoner_node)
  ament_target_dependencies(test_node rclcpp sensor_msgs)

  ament_add_gtest(test_rviz_formatting
    test/test_rviz_formatting.cpp
  )
  target_include_directories(test_rviz_formatting PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_rviz_formatting
    edge_vlm_rviz_formatting
    "${cpp_typesupport_target}"
  )
  ament_target_dependencies(test_rviz_formatting rclcpp)

  # IPC protocol tests (hardware-independent)
  ament_add_gtest(test_ipc_protocol
    test/test_ipc_protocol.cpp
  )
  target_include_directories(test_ipc_protocol PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )
  target_link_libraries(test_ipc_protocol ${OpenCV_LIBRARIES})

  # IPC backend integration tests with a fake Unix socket worker (no GPU)
  ament_add_gtest(test_ipc_inference_backend
    test/test_ipc_inference_backend.cpp
    src/ipc_inference_backend.cpp
  )
  target_include_directories(test_ipc_inference_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )
  target_link_libraries(test_ipc_inference_backend ${OpenCV_LIBRARIES})

  # InferenceWatchdog component tests (no GPU, no TensorRT)
  ament_add_gtest(test_inference_watchdog
    test/test_inference_watchdog.cpp
  )
  target_include_directories(test_inference_watchdog PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  ament_add_gtest(test_latest_value_stage
    test/test_latest_value_stage.cpp
  )
  target_include_directories(test_latest_value_stage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  ament_add_gtest(test_iou_tracker
    test/test_iou_tracker.cpp
  )
  target_include_directories(test_iou_tracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  ament_add_gtest(test_tracked_observation_adapter
    test/test_tracked_observation_adapter.cpp
  )
  target_include_directories(test_tracked_observation_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_tracked_observation_adapter
    tracked_observation_adapter
    "${cpp_typesupport_target}"
  )
  ament_target_dependencies(test_tracked_observation_adapter
    message_filters
    rclcpp
    sensor_msgs
    vision_msgs
  )

  if(BUILD_YOLO_ADAPTER)
    ament_add_gtest(test_yolo_detection2d_adapter
      test/test_yolo_detection2d_adapter.cpp
    )
    target_include_directories(test_yolo_detection2d_adapter PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_yolo_detection2d_adapter
      yolo_detection2d_adapter
    )
    ament_target_dependencies(test_yolo_detection2d_adapter
      rclcpp
      vision_msgs
      yolo_msgs
    )
  endif()

  # Shell-level unit tests for the PID-match helpers and PGID isolation safety
  # checks used in the worker-recovery hardware validation script.
  # Runs without ROS, hardware, or network access.
  find_program(BASH_EXECUTABLE bash REQUIRED)
  ament_add_test(test_worker_recovery_helpers
    COMMAND "${BASH_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test/test_worker_recovery_helpers.sh"
    TIMEOUT 30
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
  )
endif()

ament_export_targets(vlm_reasoner_nodeTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp rcl_interfaces sensor_msgs std_msgs builtin_interfaces image_transport message_filters vision_msgs)
ament_package()
