#pragma once
#include <vector>
#include <stdexcept>
#include "stream_buffer.hpp"
/**
 * @brief 核心模块 对外 API
 *负责
 *       buffer 池管理
 *       stream / event 生命周期
 *       buffer 状态机（FREE / INFLIGHT / COMPLETED）
 *       acquire / submit / fetch / release 的唯一合法路径
 * 不负责
 *       kernel 逻辑
 *       memcpy 具体内容
 *       stage 定义
 *      benchmark timing
 * 支持
 *       double buffer
 *       triple buffer
 *       N-buffer
 *       burst pipeline
FramePipeline → Thread-safe MPSC 多生产者单消费者
        多生产者单消费者（Multi-Producer Single-Consumer）线程安全队列
        多个摄像头或视频流同时输入，单个GPU进行处理
        高效并发，避免锁竞争
FramePipeline → CUDA Graph backend    使用CUDA图优化GPU执行
        减少GPU内核启动开销
        启动开销从微秒级降到纳秒级
        适合流水线化的固定计算模式
        在NVIDIA Ampere+架构上效果显著
FramePipeline → Persistent kernel   持久化内核技术
        避免重复的内核启动开销
        消除内核启动延迟
        GPU持续运行，利用率高
        适合实时视频处理
FramePipeline → video/inference runtime  视频编解码与AI推理运行时集成
        智能视频分析：多摄像头实时监控
        自动驾驶：多传感器融合处理
        直播/视频会议：实时美颜、虚拟背景
        医疗影像：实时超声/MRI分析
        工业检测：高速生产线质检

      技术	                解决的问题	                性能提升
    MPSC队列	            多输入同步	           减少锁竞争，提高吞吐量
    CUDA图	                内核启动开销	            降低延迟10-100倍
    持久内核	            内核调度开销	            稳定低延迟
    硬件编解码	            CPU-GPU传输	            零拷贝，高帧率
 * @tparam T
 */
template <typename T>
class FramePipeline
{
public:
    explicit FramePipeline(size_t frame_elements, int buffers = 3);

    // ===== Producer side =====
    stream_buffer<T> &acquire();     // 只能拿 FREE
    void submit(stream_buffer<T> &); // FREE → INFLIGHT

    // ===== Consumer side =====
    stream_buffer<T> *try_fetch();    // INFLIGHT → COMPLETED
    void release(stream_buffer<T> &); // COMPLETED → FREE

    int capacity() const noexcept { return buffers_.size(); }

private:
    std::vector<stream_buffer<T>> buffers_;
    size_t write_cursor_{0};
};
