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

# CUDA support - check for compiler first
find_program(CUDA_NVCC_EXECUTABLE nvcc PATHS
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin"
    ENV CUDA_PATH
    ENV CUDA_BIN_PATH
    PATH_SUFFIXES bin
)

if(CUDA_NVCC_EXECUTABLE)
    # Extract CUDA toolkit root from nvcc path
    get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_NVCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_TOOLKIT_ROOT}" DIRECTORY)

    # Convert to Windows path format
    file(TO_CMAKE_PATH "${CUDA_TOOLKIT_ROOT}" CUDA_TOOLKIT_ROOT)
    
    # Set CUDA paths explicitly with FORCE
    set(CMAKE_CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT}" CACHE PATH "CUDA Toolkit root directory" FORCE)
    set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
    set(CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR "${CUDA_TOOLKIT_ROOT}" CACHE PATH "CUDA toolkit directory for Visual Studio integration" FORCE)

    message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT}")
    
    # Try to find CUDAToolkit BEFORE enable_language (Option 3 approach)
    # This might help CMake discover the toolset
    find_package(CUDAToolkit QUIET PATHS "${CUDA_TOOLKIT_ROOT}" NO_DEFAULT_PATH)
    
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDAToolkit found before enable_language")
    else()
        message(STATUS "CUDAToolkit not found yet, will try after enable_language")
    endif()
    
    # Now enable CUDA language
    enable_language(CUDA)
    
    # Find CUDA toolkit again (required this time)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set CUDA properties for the kernel file
    set_property(SOURCE src/cuda-kernel.cu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(SOURCE src/cuda-kernel.cu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    # Set CUDA architecture
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures to build for")
    endif()
    
    # Link CUDA libraries
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE 
        CUDA::cudart
        CUDA::cuda_driver
    )
    
    # Add CUDA kernel source
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/cuda-kernel.cu)
    
    message(STATUS "CUDA found - CUDA features enabled")
else()
    message(WARNING "CUDA compiler (nvcc) not found - building without CUDA support")
    # Add compile definition to disable CUDA code
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE DISABLE_CUDA)
endif()

# Link D3D11 (required for CUDA-D3D11 interop, but also useful without CUDA)
if(WIN32)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE d3d11)
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

# Add main plugin source (always included)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/plugin-main.c)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})