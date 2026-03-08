# GPU 图像卷积加速库（gpu_conv_lib）

一个基于 CUDA 的高性能 GPU 图像卷积加速库，同时提供 CPU 与 GPU
Benchmark，用于性能分析与回归检测。

本项目不仅关注 **性能提升**，还关注：

-   CUDA 内存优化策略分析
-   CPU vs GPU 性能对比
-   工程级性能回归检测（Performance Regression）
-   CUDA 不同内存模型的真实行为研究

------------------------------------------------------------------------

# 项目简介

在现代图像处理系统中，大量操作都依赖卷积计算，例如：

-   图像模糊
-   边缘检测
-   锐化
-   特征提取

CPU 在处理大规模图像卷积时往往性能受限，而 GPU
拥有大规模并行计算能力，非常适合此类任务。

本项目实现：

-   CPU 卷积实现
-   CUDA 高性能卷积实现
-   Benchmark 性能评测框架
-   CPU vs GPU 对比测试
-   图像查看工具

目标不仅是 **加速计算**，还包括理解 **真实 GPU 硬件行为**。

------------------------------------------------------------------------

# 功能特性

-   CUDA GPU 加速卷积
-   多种卷积算子
-   CPU 与 GPU Benchmark
-   Shared Memory 优化实验
-   Constant Memory 权重存储
-   图像查看器（SDL3）
-   跨平台支持

支持卷积类型：

-   Gaussian Blur（高斯模糊）
-   Mean Blur（均值滤波）
-   Sobel（边缘检测）
-   Laplacian（拉普拉斯）
-   Sharpen（锐化）

------------------------------------------------------------------------

# 编译环境

需要以下环境：

-   CMake ≥ 3.15
-   CUDA Toolkit ≥ 10.0
-   C++17 编译器

可选组件：

-   SDL3
-   SDL3_image

------------------------------------------------------------------------

# 编译方法

## Linux

``` bash
mkdir build
cd build

cmake .. -DBUILD_SAMPLES=ON -DBUILD_IMAGE_VIEWER=ON

make -j$(nproc)
```

## Windows

``` bash
mkdir build
cd build

cmake .. ^
-G "Visual Studio 17 2022" ^
-A x64 ^
-DBUILD_SAMPLES=ON ^
-DBUILD_IMAGE_VIEWER=ON

cmake --build . --config Release
```

------------------------------------------------------------------------

# CPU 卷积 Benchmark

CPU Benchmark 的目标是 **检测性能是否发生退化**。

CPU 卷积在现代处理器上通常属于：

**Memory Bound（内存带宽受限）任务**。

因此性能评价采用：

**吞吐量（MP/s）**

而不是固定执行时间。

MP/s 表示：

每秒处理多少 **百万像素**。

------------------------------------------------------------------------

# CPU 性能测试结果

  卷积核          分辨率      时间      吞吐量
  --------------- ----------- --------- -----------
  Gaussian 3×3    1024×1024   23.3 ms   45 MP/s
  Gaussian 5×5    1024×1024   53.8 ms   19.5 MP/s
  Mean 3×3        1024×1024   22.7 ms   46 MP/s
  Laplacian 3×3   1024×1024   22.0 ms   47 MP/s

合理性能区间：

  Kernel Size   吞吐量
  ------------- ---------------
  3×3           40 -- 55 MP/s
  5×5           15 -- 25 MP/s
  ≥7×7          5 -- 15 MP/s

CI 性能阈值示例：

``` cpp
if (ksize <= 3)
    REQUIRE(throughput >= 35);
else if (ksize <= 5)
    REQUIRE(throughput >= 15);
else
    REQUIRE(throughput >= 5);
```

该阈值用于 **检测性能回退**，而不是比较极限性能。

------------------------------------------------------------------------

# CUDA 卷积 Benchmark

CUDA Benchmark 用于分析不同 GPU 内存策略的性能差异：

-   Global Memory
-   Shared Memory
-   Constant Memory
-   CUDA Streams

测试分辨率：

1920 × 1080

数据类型：

float

------------------------------------------------------------------------

# Sobel Benchmark

  内存策略            Kernel 时间   吞吐量
  ------------------- ------------- ----------------
  Global Memory       0.139 ms      14.86 GPixel/s
  Shared + Constant   0.478 ms      4.33 GPixel/s

结论：

对于 **Sobel 3×3**：

Global Memory 明显更快。

原因：

-   计算量很小
-   访存模式连续
-   L1 Cache 已经完成数据复用

Shared Memory 反而引入额外开销：

-   halo 加载
-   global → shared 拷贝
-   \_\_syncthreads() 同步

------------------------------------------------------------------------

# Gaussian Benchmark

  Kernel Size   Global      Shared      更优方案
  ------------- ----------- ----------- ----------
  3×3           0.0607 ms   0.0758 ms   Global
  5×5           0.1897 ms   0.1503 ms   Shared
  7×7           0.2978 ms   0.1454 ms   Shared
  9×9           0.4919 ms   0.2366 ms   Shared

性能拐点：

Kernel Size ≈ 5

------------------------------------------------------------------------

# CUDA 内存优化策略

工程实践中可以采用如下规则：

``` cpp
if (filter == SOBEL && kSize == 3)
    useGlobalKernel();
else if (filter == GAUSSIAN && kSize < 5)
    useGlobalKernel();
else
    useSharedKernel();
```

关键原则：

**Shared Memory 不是默认优化方案。**

只有在 **数据复用明显增加时** 才会带来性能收益。

------------------------------------------------------------------------

# 进一步优化方向

## Separable Gaussian

将复杂度

O(k²)

降低为

O(2k)

对于大卷积核（k ≥ 7）性能提升明显。

------------------------------------------------------------------------

## Profiling 分析

推荐工具：

-   NVIDIA Nsight Compute
-   NVIDIA Nsight Systems
-   Intel VTune

重点关注指标：

-   L1 Cache 命中率
-   DRAM 带宽
-   Shared Memory 访问次数
-   同步指令数量

------------------------------------------------------------------------

# 性能总结

CPU

3×3 卷积：约 45 MP/s

5×5 卷积：约 18 MP/s

GPU

Sobel 3×3 → Global Memory 最优

Gaussian ≥5 → Shared Memory 更优

核心结论：

Sobel 3×3 是 **L1 Cache 的优势场景**。

Gaussian 大卷积核才是 **Shared Memory 的主战场**。

------------------------------------------------------------------------

# License

MIT License
