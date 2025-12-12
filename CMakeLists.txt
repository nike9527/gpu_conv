cmake_minimum_required(VERSION 3.18)
project(gpu_conv_lib LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#--------------------------
# Options
# BUILD_SAMPLES：     是否编译生成可执行文件
# BUILD_SAMPLES：     是否使用OPENMP
# BUILD_IMAGE_VIEWER：图片查看器工具
#--------------------------
option(BUILD_SAMPLES "Build sample executables" ON)
option(USE_OPENMP "Use OpenMP for CPU baseline" ON)
option(BUILD_IMAGE_VIEWER "Build the image viewer tool" ON)
#--------------------------
# CUDA
#--------------------------
find_package(CUDAToolkit REQUIRED)
set(CMAKE_BUILD_TYPE Debug)
#--------------------------
# 配置输出目录
#--------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#--------------------------
# MSVC Runtime (统一 /MD /MDd)
#--------------------------
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()


#--------------------------
# gconv库
#--------------------------
add_library(gconv STATIC
    src/convolution_cpu.cpp
    src/image.cpp
    src/kernel.cpp
    cuda/convolution_gpu.cu
    cuda/convolution_global.cu
    cuda/convolution_shared.cu
)
set_target_properties(gconv PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(gconv PUBLIC include)
target_link_libraries(gconv PUBLIC CUDA::cudart)
target_compile_definitions(gconv PRIVATE GCONV_EXPORTS)

#--------------------------
# OpenMP
#--------------------------
if(USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gconv PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

#--------------------------
# SDL3 和 SDL3_image 图片查看器
#--------------------------
if(BUILD_IMAGE_VIEWER)
    set(SDL3_DIR "${CMAKE_SOURCE_DIR}/external/SDL3")
    set(SDL3_IMAGE_DIR "${CMAKE_SOURCE_DIR}/external/SDL3_image")
    set(IMAGE_VIEWER_DIR "${CMAKE_SOURCE_DIR}/utils/image_viewer")
    #关闭测试 & 安装
    set(SDL_SHARED ON  CACHE BOOL "" FORCE)
    set(SDL_STATIC ON  CACHE BOOL "" FORCE)
    set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
    set(SDL_TEST OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    #关闭一切外部依赖
    set(SDLIMAGE_AVIF OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_WEBP OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_JXL  OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_TIF  OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_HDR  OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_QOI  OFF CACHE BOOL "" FORCE)
    # 关闭 samples/tests/install
    set(SDLIMAGE_TESTS OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_SAMPLES OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_INSTALL OFF CACHE BOOL "" FORCE)
    # 启用 PNG + JPG
    set(SDLIMAGE_PNG ON CACHE BOOL "" FORCE)
    set(SDLIMAGE_JPG ON CACHE BOOL "" FORCE)
    if(EXISTS "${SDL3_DIR}/CMakeLists.txt")
        add_subdirectory(${SDL3_DIR})
    endif()

    if(EXISTS "${SDL3_IMAGE_DIR}/CMakeLists.txt")
        add_subdirectory(${SDL3_IMAGE_DIR})
    endif()

    if(EXISTS "${IMAGE_VIEWER_DIR}/CMakeLists.txt")
      add_subdirectory(${IMAGE_VIEWER_DIR})
    endif()
endif()
#--------------------------
# 部分程序示例
#--------------------------
if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()