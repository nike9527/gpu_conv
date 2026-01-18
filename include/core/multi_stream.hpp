#pragma once
#include "stream_buffer.hpp"
#include "cuda/cuda_memory.hpp"
#include <memory>
#include <mutex>
#include <condition_variable>
template <typename T, int N = 3>
/**
 * @brief
 “已完成但仍占用”
CUDA event 只保证 GPU 不再使用 buffer
buffer 是否能复用，取决于系统是否还需要它
 *
 */
class multi_stream
{
public:
    explicit multi_stream(size_t max_elements)
    {
        for (auto &buf : buffers_)
            buf.allocate(max_elements);
    }
    // 阻塞，直到拿到一个可写 buffer
    stream_buffer<T> &acquire();
    // 提交 GPU 工作（进入 inflight）
    void submit(stream_buffer<T> &buf);
    // 非阻塞获取一个“已完成但仍占用”的 buffer
    stream_buffer<T> *try_fetch();
    // 消费完成，显式释放
    void release(stream_buffer<T> &buf);

    int inflight() const noexcept { return inflight_; }

private:
    stream_buffer<T> buffers_[N];
    int write_idx_ = 0;
    int read_idx_ = 0;
    // 执行中的数量
    int inflight_ = 0;
};
/**
 * @brief 优先返回 FREE buffer
 *
 * @return template <typename T, int N>&
for each buffer:
    if !busy → return
if none:
    wait any event → free → return
*/
// TODO
template <typename T, int N>
stream_buffer<T> &multi_stream<T, N>::acquire()
{
    // 1. 先尝试找到一个空闲 buffer
    for (int i = 0; i < N; ++i)
    {
        auto &buf = buffers_[write_idx_];
        if (buf.is_free())
        {
            write_idx_ = (write_idx_ + 1) % N;
            return buf;
        }
        // GPU 已完成，回收
        // if (cudaEventQuery(buffers_[idx].event()) == cudaSuccess)
        // {
        //     buffers_[idx].mark_free();
        //     inflight_--;
        //     write_idx_ = (idx + 1) % N;
        //     return buffers_[idx];
        // }
        write_idx_ = (write_idx_ + 1) % N;
    }
    // 采用条件变量通知
    //  std::unique_lock<std::mutex> lock(syncData.mtx);
    //  syncData.cv.wait(lock, [&]()
    //                   { return syncData.completed_count == NUM_STREAMS; });

    // // 2. 全部 busy，等待最早完成的那个（通常是 read_idx_）
    // auto &buf = buffers_[read_idx_];

    // CUDA_CHECK(cudaEventSynchronize(buf.event()));
    // buf.mark_free();
    // inflight_--;

    // write_idx_ = (read_idx_ + 1) % N;
    // return buf;
    //=========================上面注释调整到下面================================
    // 没有 FREE buffer：必须等待一个 COMPLETED 被 release
    // 这里可以选择阻塞 / spin / yield
    throw std::runtime_error("no free buffer available");
}

template <typename T, int N>
void multi_stream<T, N>::submit(stream_buffer<T> &buf)
{
    // 必须是空闲 buffer
    // assert(!buf.busy());
    // CUDA_CHECK(cudaEventRecord(buf.event(), buf.stream()));
    // buf.mark_busy();
    // inflight_++;
    //=========================上面注释调整到下面================================
    if (!buf.is_free())
    {
#ifndef NDEBUG
        // 调试版本：详细异常
        throw std::logic_error("Cannot submit free buffer.");
#else
        // 发布版本：快速失败
        return;
#endif
    }
    CUDA_CHECK(cudaEventRecord(buf.event(), buf.stream()));
    buf.mark_inflight();
    inflight_++;
}
// 支持乱序完成
/**
 * @brief
 *
 * @tparam T
 * @tparam N
 * @return stream_buffer<T>*
 for each buffer:
    if busy && event ready → return
 */
// TODO
template <typename T, int N>
stream_buffer<T> *multi_stream<T, N>::try_fetch()
{
    // for (int i = 0; i < N; ++i)
    // {
    //     auto &buf = buffers_[read_idx_];
    //     if (!buf.busy())
    //         continue;
    //     auto err = cudaEventQuery(buf.event());
    //     if (err == cudaErrorNotReady)
    //         continue;
    //     CUDA_CHECK(err);
    //     return &buf;
    // }
    //=========================上面注释调整到下面================================
    for (int i = 0; i < N; ++i)
    {
        auto &buf = buffers_[read_idx_];

        if (buf.is_inflight())
        {
            if (cudaEventQuery(buf.event()) == cudaSuccess)
            {
                buf.mark_completed();
                read_idx_ = (read_idx_ + 1) % N;
                return &buf;
            }
        }

        read_idx_ = (read_idx_ + 1) % N;
    }
    return nullptr;
}

template <typename T, int N>
void multi_stream<T, N>::release(stream_buffer<T> &buf)
{
    // assert(buf.busy());
    // buf.mark_free();
    // inflight_--;
    //=========================上面注释调整到下面================================
    if (!buf.is_completed())
        throw std::logic_error("release non-completed buffer");
    inflight_--;
    buf.mark_free();
}