cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

# CUDA headers for TensorRT (cuda_runtime_api.h, cuda_fp16.h, etc.)
if(DEFINED ENV{CUDA_PATH})
    message(STATUS "Using CUDA from $ENV{CUDA_PATH}")
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/include")

    # Link CUDA runtime library (for runtime D3D11 interop)
    find_library(CUDA_RUNTIME_LIBRARY
        NAMES cudart
        PATHS "$ENV{CUDA_PATH}/lib/x64" "$ENV{CUDA_PATH}/lib"
        NO_DEFAULT_PATH
    )
    if(CUDA_RUNTIME_LIBRARY)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${CUDA_RUNTIME_LIBRARY})
        message(STATUS "CUDA runtime library found: ${CUDA_RUNTIME_LIBRARY}")
    else()
        message(WARNING "CUDA runtime library not found - D3D11 interop may not work")
    endif()
else()
    # Fallback: hard-coded path, adjust version as needed
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
    )
    # Try to find cudart in fallback location
    find_library(CUDA_RUNTIME_LIBRARY
        NAMES cudart
        PATHS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64"
        NO_DEFAULT_PATH
    )
    if(CUDA_RUNTIME_LIBRARY)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${CUDA_RUNTIME_LIBRARY})
    endif()
endif()

# Prevent HLSL files from being compiled at build time
# They are compiled at runtime using D3DCompileFromFile
file(GLOB HLSL_FILES "data/*.hlsl")
foreach(HLSL_FILE ${HLSL_FILES})
    set_source_files_properties(${HLSL_FILE} PROPERTIES
        VS_TOOL_OVERRIDE "None"
        HEADER_FILE_ONLY TRUE
    )
endforeach()

# Prevent HLSL files from being compiled at build time
# They are compiled at runtime using D3DCompileFromFile
file(GLOB HLSL_FILES "data/*.hlsl")
foreach(HLSL_FILE ${HLSL_FILES})
    set_source_files_properties(${HLSL_FILE} PROPERTIES
        VS_TOOL_OVERRIDE "None"
        HEADER_FILE_ONLY TRUE
    )
endforeach()


# Link D3D11 and D3DCompiler (required for CUDA-D3D11 interop and shader compilation)
if(WIN32)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE d3d11 d3dcompiler)
    
    # Add Windows-specific compile definitions
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        NOMINMAX  # Prevent Windows.h from defining min/max macros
        WIN32_LEAN_AND_MEAN  # Reduce Windows.h bloat
    )
endif()

# TensorRT support
# Try multiple common installation paths
set(TENSORRT_POSSIBLE_ROOTS
    "C:/TensorRT-10.14.1.48"
    "C:/TensorRT-10.1"
    "C:/TensorRT-10.0"
    "$ENV{TENSORRT_ROOT}"
    "$ENV{TENSORRT_PATH}"
)

find_path(TENSORRT_INCLUDE_DIR
    NvInfer.h
    PATHS ${TENSORRT_POSSIBLE_ROOTS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

# If not found in specific paths, try default system paths
if(NOT TENSORRT_INCLUDE_DIR)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATH_SUFFIXES include)
endif()

find_library(TENSORRT_LIBRARY
    NAMES nvinfer nvinfer_10
    PATHS ${TENSORRT_POSSIBLE_ROOTS}
    PATH_SUFFIXES lib lib/x64
    NO_DEFAULT_PATH
)

# If not found in specific paths, try default system paths
if(NOT TENSORRT_LIBRARY)
    find_library(TENSORRT_LIBRARY NAMES nvinfer nvinfer_10 PATH_SUFFIXES lib lib64)
endif()

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${TENSORRT_LIBRARY})
    message(STATUS "TensorRT found:")
    message(STATUS "  Include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  Library: ${TENSORRT_LIBRARY}")
else()
    message(FATAL_ERROR "TensorRT not found - plugin requires TensorRT to build")
    message(STATUS "  Searched in: ${TENSORRT_POSSIBLE_ROOTS}")
    message(STATUS "  Set TENSORRT_ROOT environment variable to TensorRT installation directory")
endif()

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

# Set C++ standard for TensorRT runner (C++17 required for TensorRT 10.x)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# Add main plugin source and TensorRT runner (always included)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE 
    src/plugin-main.cpp
    src/trt_runner.cpp
)

# Ensure these sources are compiled as C++
set_source_files_properties(
    src/trt_runner.cpp
    src/plugin-main.cpp
    PROPERTIES LANGUAGE CXX
)

# On MSVC, disable deprecation warning C4996 coming from TensorRT headers
if(MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /wd4996)
endif()

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})