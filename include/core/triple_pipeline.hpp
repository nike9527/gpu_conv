#pragma once
#include "stream_buffer.hpp"
#include "cuda/cuda_memory.hpp"
#include <array>
#include <cassert>
//
/**
 * @brief
 TODO acquire / submit / fetch 做成 lock-free MPSC 版本
 TODO 异步提交版本，同一个stream可以多次提交，不用等待FREE
 * @tparam SlotT
 * @tparam N
 */
template <typename SlotT, int N = 3>
class triple_pipeline
{
public:
    explicit triple_pipeline() = default;
    // explicit triple_pipeline(size_t max_elements)
    // {
    //     for (auto &buf : slots_)
    //         buf.allocate(max_elements);
    // }
    // 阻塞，直到拿到一个可写 buffer
    SlotT &acquire();
    // 提交 GPU 工作（进入 inflight）
    void submit(SlotT &buf);
    // 非阻塞获取一个“已完成但仍占用”的 buffer
    SlotT *try_fetch();
    // 消费完成，显式释放
    void release(SlotT &buf);

    int inflight() const noexcept { return inflight_; }

private:
    SlotT slots_[N];
    int write_idx_ = 0;
    int read_idx_ = 0;
    // 执行中的数量
    int inflight_ = 0;
};
/**
 * @brief 优先返回 FREE buffer
 * @return SlotT&
 */
// TODO
template <typename SlotT, int N>
SlotT &triple_pipeline<SlotT, N>::acquire()
{
    // 1. 先尝试找到一个空闲 buffer
    for (int i = 0; i < N; ++i)
    {
        auto &buf = slots_[write_idx_];
        if (buf.is_free())
        {
            write_idx_ = (write_idx_ + 1) % N;
            return buf;
        }
        write_idx_ = (write_idx_ + 1) % N;
    }
    // 没有 FREE buffer：必须等待一个 COMPLETED 被 release
    // 这里可以选择阻塞 / spin / yield
    throw nullptr;
}

template <typename SlotT, int N>
void triple_pipeline<SlotT, N>::submit(SlotT &buf)
{
    if (!buf.is_free())
    {
#ifndef NDEBUG
        throw std::logic_error("Cannot submit non-free buffer.");
#else
        return;
#endif
    }
    CUDA_CHECK(cudaEventRecord(buf.event(), buf.stream()));
    buf.mark_inflight();
    inflight_++;
}
// TODO
template <typename SlotT, int N>
SlotT *triple_pipeline<SlotT, N>::try_fetch()
{
    for (int i = 0; i < N; ++i)
    {
        auto &buf = slots_[read_idx_];

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

template <typename SlotT, int N>
void triple_pipeline<SlotT, N>::release(SlotT &buf)
{
    if (!buf.is_completed())
        return;
    inflight_--;
    buf.mark_free();
}