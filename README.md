# GPU 图像卷积加速库 (gpu_conv_lib)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 简介

GPU图像卷积加速库是一个高性能的图像处理库，利用GPU并行计算能力加速卷积操作。支持后端（CUDA/）和多种卷积核类型。

## 功能特性

- ✅ GPU加速的卷积操作（最高可达CPU的100倍速度）
- ✅ 支持CUDA后端
- ✅ 多种内置卷积核（高斯模糊、边缘检测、锐化等）
- ✅ 支持自定义卷积核
- ✅ 多图像格式支持（JPEG、PNG、BMP等）
- ✅ 跨平台支持（Windows、Linux、macOS）
- ✅ 简单易用的API接口
- ✅CPU 对比基准（OpenMP 可选）
- ✅Image Viewer（SDL3 + SDL3_image）
- ✅包含示例程序 samples_cpu_vs_gpu

## 目录结构
- bin/ # 可执行文件输出
- lib/ # 静态库输出
- include/ # 公共头文件
- src/ # GPU 卷积核心代码
- samples/ # 示例程序
- utils/image_viewer/ # 图像查看器库
- external/SDL3 # SDL3源码
- external/SDL3_image # SDL3_image源码


## 快速开始

### 前提条件

- CMake 3.15+
- CUDA Toolkit 10.0+（可选，用于CUDA后端）
- OpenCL 1.2+（可选，用于OpenCL后端）
- C++17兼容编译器

## 编译方法

### Windows (MSVC)
- mkdir build
- cd build
- cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_SAMPLES=ON -DBUILD_IMAGE_VIEWER=ON
- cmake --build . --config Release

### Linux (GCC)
- mkdir build
- cd build
- cmake .. -DBUILD_SAMPLES=ON -DBUILD_IMAGE_VIEWER=ON
- make -j$(nproc)
### 构建

```bash
# Linux/macOS
chmod +x build.sh
./build.sh

# Windows
build.bat