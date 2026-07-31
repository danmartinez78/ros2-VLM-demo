cmake_minimum_required(VERSION 3.20)
project(cosmos_ros2_video_reasoner LANGUAGES C CXX)

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
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ── Custom message ────────────────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VisionReasoningResult.msg"
  DEPENDENCIES std_msgs builtin_interfaces
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
add_library(cosmos_reasoner_node SHARED
  src/cosmos_reasoner_node.cpp
)

target_include_directories(cosmos_reasoner_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(cosmos_reasoner_node
  rclcpp
  rcl_interfaces
  sensor_msgs
  std_msgs
  builtin_interfaces
  image_transport
)

target_link_libraries(cosmos_reasoner_node
  ${OpenCV_LIBRARIES}
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(cosmos_reasoner_node "${cpp_typesupport_target}")

# ── TensorRT backend library (hardware-dependent) ────────────────────────────
if(BUILD_TENSORRT_BACKEND)
  add_library(cosmos_trt_backend SHARED
    src/tensorrt_edge_llm_backend.cpp
  )

  target_include_directories(cosmos_trt_backend PUBLIC
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

  target_link_libraries(cosmos_trt_backend
    cosmos_reasoner_node
    ${EDGELLM_CORE_LIB}
    ${CUTE_DSL_STATIC_LIB}
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    CUDA::cudart
    CUDA::cuda_driver
    dl
    ${OpenCV_LIBRARIES}
  )

  ament_target_dependencies(cosmos_trt_backend rclcpp rcl_interfaces)
  if(EXISTS "${CUTE_DSL_RUNTIME_LIB}")
    target_link_libraries(cosmos_trt_backend
      ${CUTE_DSL_RUNTIME_LIB})
  endif()

  set_target_properties(cosmos_trt_backend PROPERTIES
    CUDA_ARCHITECTURES "${TENSORRT_EDGE_LLM_CUDA_ARCHITECTURES}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INSTALL_RPATH_USE_LINK_PATH TRUE)

  # ── Direct-linked diagnostic executable ───────────────────────────────
  # Compiles the backend implementation into the final executable instead of
  # consuming libcosmos_trt_backend.so. This isolates CUDA device-link and
  # shared-library behavior from the backend implementation itself.
  add_executable(cosmos_backend_direct_smoke
    src/backend_direct_smoke.cpp
    src/tensorrt_edge_llm_backend.cpp
  )
  target_include_directories(cosmos_backend_direct_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_EDGE_LLM_ROOT}/cpp
    ${TENSORRT_EDGE_LLM_ROOT}/examples
    ${TRT_INCLUDE_DIR}
    ${_NLOHMANN_JSON_INCLUDE}
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/stb
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/miniaudio
    ${TENSORRT_EDGE_LLM_ROOT}/examples/multimodal
  )
  ament_target_dependencies(cosmos_backend_direct_smoke rclcpp)
  target_link_libraries(cosmos_backend_direct_smoke
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
    target_link_libraries(cosmos_backend_direct_smoke
      ${CUTE_DSL_RUNTIME_LIB})
  endif()
  set_target_properties(cosmos_backend_direct_smoke PROPERTIES
    CUDA_ARCHITECTURES "${TENSORRT_EDGE_LLM_CUDA_ARCHITECTURES}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INSTALL_RPATH_USE_LINK_PATH TRUE)

  # ── ROS-free persistent inference worker ──────────────────────────────
  add_executable(cosmos_inference_worker
    src/inference_worker.cpp
    src/tensorrt_edge_llm_backend.cpp
  )
  target_include_directories(cosmos_inference_worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_EDGE_LLM_ROOT}/cpp
    ${TENSORRT_EDGE_LLM_ROOT}/examples
    ${TRT_INCLUDE_DIR}
    ${_NLOHMANN_JSON_INCLUDE}
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/stb
    ${TENSORRT_EDGE_LLM_ROOT}/3rdParty/miniaudio
    ${TENSORRT_EDGE_LLM_ROOT}/examples/multimodal
  )
  target_link_libraries(cosmos_inference_worker
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
    target_link_libraries(cosmos_inference_worker ${CUTE_DSL_RUNTIME_LIB})
  endif()
  set_target_properties(cosmos_inference_worker PROPERTIES
    CUDA_ARCHITECTURES "${TENSORRT_EDGE_LLM_CUDA_ARCHITECTURES}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INSTALL_RPATH_USE_LINK_PATH TRUE)

  # ── ROS executable: IPC only, never links Edge-LLM ─────────────────────
  add_library(cosmos_ipc_backend STATIC src/ipc_inference_backend.cpp)
  target_include_directories(cosmos_ipc_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(cosmos_ipc_backend ${OpenCV_LIBRARIES})

  add_executable(cosmos_reasoner src/main.cpp)
  target_include_directories(cosmos_reasoner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(cosmos_reasoner
    cosmos_ipc_backend
    cosmos_reasoner_node)
  ament_target_dependencies(cosmos_reasoner rclcpp)

  install(TARGETS cosmos_trt_backend cosmos_reasoner cosmos_backend_direct_smoke cosmos_inference_worker cosmos_ipc_backend
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )

  # Set RPATH so the plugin library is found at runtime without LD_LIBRARY_PATH
  set_target_properties(cosmos_reasoner PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
  )
else()
  message(STATUS
    "TensorRT backend disabled. "
    "Building node library and tests only (no GPU required).")
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS cosmos_reasoner_node
  EXPORT cosmos_reasoner_nodeTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
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
  target_link_libraries(test_frame_sampling cosmos_reasoner_node)
  ament_target_dependencies(test_frame_sampling rclcpp sensor_msgs)

  # Node integration tests (no GPU)
  ament_add_gtest(test_node
    test/test_node.cpp
  )
  target_include_directories(test_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )
  target_link_libraries(test_node cosmos_reasoner_node)
  ament_target_dependencies(test_node rclcpp sensor_msgs)
endif()

ament_export_targets(cosmos_reasoner_nodeTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp rcl_interfaces sensor_msgs std_msgs builtin_interfaces image_transport)
ament_package()
