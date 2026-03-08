# CUDA Convolution 性能深度分析（Perf Analysis）

本文档从 **GPU 硬件行为视角** 出发，对 Sobel 与 Gaussian 卷积在不同 kernel size 下，
**Global / L1 Cache / Shared Memory / Constant Memory** 的真实性能表现进行系统分析。

目标读者：
- 具备 CUDA 基础
- 正在做 **性能工程 / Benchmark / Kernel 优化** 的开发者

---

## 一、硬件背景速览（以现代 NVIDIA GPU 为例）

### SM 内存层级（延迟由低到高）

| 层级 | 典型延迟 | 说明 |
|----|----|----|
| Register | ~1 cycle | 线程私有 |
| Shared Memory | ~20–30 cycles | 显式管理，确定性 |
| L1 Cache（命中） | ~20–30 cycles | 自动缓存，不确定性 |
| L2 Cache | ~200 cycles | 跨 SM |
| Global Memory | 400–800 cycles | DRAM |

> **Shared Memory 与 L1 Cache 物理上是同一块 SM 内 SRAM，只是使用语义不同。**

---

## 二、Sobel 3×3：为什么 Global + L1 完胜 Shared

### 1️⃣ Sobel 3×3 的计算特征

- 每像素访问 9 个邻域值
- 访问模式线性、连续
- warp 内天然合并访问

```text
Global kernel ≈ 9 loads / pixel
Shared kernel ≈ 1 global + halo + sync + 9 shared loads
```

---

### 2️⃣ L1 Cache 在 Sobel 中的“天然优势”

- Cache line 一次拉取可覆盖多个邻域像素
- 相邻线程访问高度重叠
- cache 命中率接近 100%

> **Global kernel 已经“白嫖”了 Shared 的复用效果**

---

### 3️⃣ Shared 版本的反优化来源

- 额外的 global → shared copy
- halo 线程占比高
- `__syncthreads()` 同步成本

> 对于算术强度极低的 Sobel 3×3，这些开销无法被摊销

---

## 三、Gaussian：kernel size 是一切的分水岭

### 1️⃣ Global 版本的复杂度

| kSize | 每像素 global load |
|----|----|
| 3 | 9 |
| 5 | 25 |
| 7 | 49 |
| 9 | 81 |

> Global 版本是 **O(k²)** 的 global memory 压力

---

### 2️⃣ Shared 版本的复杂度变化

```text
Global → Shared : O(1)
Shared → Register : O(k²)
```

- global 带宽压力被限制在 SM 内
- shared reuse 随 kSize 增大快速放大

---

### 3️⃣ 拐点分析（kSize ≈ 5）

- kSize ≤ 3：L1 cache 足够
- kSize = 5：L1 开始 thrash
- kSize ≥ 7：Shared 完全主导

> **这不是算法问题，而是 cache working set 超出 L1 容量**

---

## 四、Constant Memory 的真实角色

- 核权重尺寸 ≤ 81 floats
- warp 广播
- constant cache 命中率 ≈ 100%

> Constant Memory 是“稳定收益”，但 **不决定 Shared 是否胜出**

---

## 五、Pipeline 对结论的影响

- Sobel：Single Stream
- Gaussian：Triple Stream

Triple Pipeline 已完全隐藏 H2D / D2H：

> **Benchmark 的 kernel 时间几乎不受 pipeline 干扰**

因此：
- 性能差异 100% 来自 kernel 设计

---

## 六、设计决策总结（工程级）

### Kernel Dispatch 决策树

```text
            +-- Sobel (3×3) --------> Global
Input ----->|
            |   Gaussian
            |      |
            |      +-- kSize < 5 ---> Global
            |      |
            |      +-- kSize ≥ 5 ---> Shared + Const
```

---

## 七、进阶方向（为下一步优化铺路）

### 1️⃣ Separable Gaussian（推荐）

- 2D 卷积 → 横向 1D + 纵向 1D
- 复杂度：O(k²) → O(2k)
- 对 kSize ≥ 7 提升巨大

---

### 2️⃣ Nsight Compute 验证指标

建议重点关注：

- `l1tex__t_sectors_hit_rate.pct`
- `dram__bytes_read.sum / pixel`
- `sm__inst_executed_barrier`

---

## 八、核心总结

>**Sobel 3×3 是 L1 Cache 的舞台，Gaussian ≥7 才是 Shared Memory 的主战场。**

这不是经验结论，而是由 **SM 内存层级与访问复杂度共同决定的硬件事实**。

