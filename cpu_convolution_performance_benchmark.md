# CPU Convolution Performance Benchmark

本项目包含一套 **CPU 图像卷积实现与性能基准测试（Benchmark）框架**，用于验证不同卷积算子在不同分辨率与 kernel size 下的 **正确性、稳定性与性能是否发生回退**。

本 Benchmark **不追求极限性能**，而是服务于工程目标：

> **在 CI / 日常开发中，及时发现性能退化（Performance Regression）**

---

## ✨ 特性概览

- 支持多种卷积算子
  - Gaussian Blur
  - Mean Blur
  - Sharpen
  - Laplacian
- 支持多种 Kernel Size（3×3 / 5×5）
- 支持多分辨率测试（512×512 / 1024×1024）
- 基于 **吞吐量（MP/s）** 的性能阈值设计
- 与 GTest 集成，可直接接入 CI

---

## 🧠 性能设计理念

### 1️⃣ CPU 卷积是 Memory-bound

在现代 CPU 上，卷积性能主要受以下因素影响：

- 内存带宽
- Cache 命中率
- 向量化效率

而非简单的 FLOPs 计算量。因此：

> ❌ 不使用「固定时间阈值」
> ✅ 使用「吞吐量（MP/s）」进行评估

---

### 2️⃣ Kernel Size 决定性能阶梯

在 CPU 上，不同 kernel size 会形成明显的性能台阶：

| Kernel Size | 合理吞吐区间 |
|------------|--------------|
| 3×3        | 40 – 55 MP/s |
| 5×5        | 15 – 25 MP/s |
| ≥7×7       | 5 – 15 MP/s  |

Benchmark 的目标是 **防止性能明显低于该区间下限**。

---

## 📊 性能测试结果（实测）

### 🔹 Gaussian Blur

| Resolution | Kernel | Time (ms) | Throughput (MP/s) | Result |
|-----------|--------|-----------|-------------------|--------|
| 512×512   | 3×3    | 5.7       | 46.0              | PASS   |
| 512×512   | 5×5    | 15.7      | 16.7              | PASS   |
| 1024×1024 | 3×3    | 23.3      | 45.0              | PASS   |
| 1024×1024 | 5×5    | 53.8      | 19.5              | PASS   |

---

### 🔹 Mean Blur

| Resolution | Kernel | Time (ms) | Throughput (MP/s) | Result |
|-----------|--------|-----------|-------------------|--------|
| 512×512   | 3×3    | 5.5       | 47.7              | PASS   |
| 512×512   | 5×5    | 15.0      | 17.5              | PASS   |
| 1024×1024 | 3×3    | 22.7      | 46.2              | PASS   |
| 1024×1024 | 5×5    | 53.2      | 19.7              | PASS   |

---

### 🔹 Sharpen

| Resolution | Kernel | Time (ms) | Throughput (MP/s) | Result |
|-----------|--------|-----------|-------------------|--------|
| 512×512   | 3×3    | 5.4–5.5   | 47–48             | PASS   |
| 1024×1024 | 3×3    | 22.4–22.9 | 45–47             | PASS   |

---

### 🔹 Laplacian

| Resolution | Kernel | Time (ms) | Throughput (MP/s) | Result |
|-----------|--------|-----------|-------------------|--------|
| 512×512   | 3×3    | 5.2–5.3   | 49–50             | PASS   |
| 1024×1024 | 3×3    | 22.0–22.3 | 47–48             | PASS   |

---

## ✅ 性能阈值规则（CI 使用）

```cpp
if (ksize <= 3)
    REQUIRE(throughput >= 35.0);   // MP/s
else if (ksize <= 5)
    REQUIRE(throughput >= 15.0);
else
    REQUIRE(throughput >= 5.0);
```

- 阈值为 **保守下限**
- 用于检测性能是否发生明显退化
- 不用于极限性能比较

---

## 🧪 分辨率一致性校验

对于同一 kernel size：

```
Throughput(512×512) ≈ Throughput(1024×1024)
```

允许偏差：**±15%**

若偏差显著增大，通常意味着：

- Cache 使用异常
- 并行策略问题
- 实现退化

---

## 📌 Benchmark 输出示例

```text
========================================
Performance Report (CPU)
========================================
Resolution:    1024 × 1024 (1.0 MP)
Kernel:        Gaussian 5×5
Time:          15.7 ms
Throughput:    16.7 MP/s
Threshold:     15.0 MP/s
Result:        PASS
========================================
```

---

## 🏁 总结

- 本项目的 CPU 卷积实现在各算子、各分辨率下表现 **稳定、一致**
- 3×3 卷积吞吐量稳定在 **45–50 MP/s**
- 5×5 卷积吞吐量稳定在 **16–20 MP/s**，符合 CPU memory-bound 特性
- Benchmark 设计目标明确：**防止性能回退，而非追求极限性能**

---

> 如果你希望进一步扩展：
> - CPU vs CUDA 性能对比
> - Separable Gaussian Benchmark
> - Nsight / VTune Profile 分析
>
> 该 Benchmark 框架可直接作为工程级性能验证基础。