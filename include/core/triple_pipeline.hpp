#pragma once
#include "stream_buffer.hpp"
#include "cuda/cuda_memory.hpp"
template <typename T, int N = 3>
class triple_pipeline
{
public:
    explicit triple_pipeline(size_t max_elements)
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
    int inflight_ = 0;
};

template <typename T, int N>
stream_buffer<T> &triple_pipeline<T, N>::acquire()
{
    // 1. 先尝试找到一个空闲 buffer
    for (int i = 0; i < N; ++i)
    {
        int idx = (write_idx_ + i) % N;
        if (!buffers_[idx].busy())
        {
            write_idx_ = (idx + 1) % N;
            return buffers_[idx];
        }
        // GPU 已完成，回收
        if (cudaEventQuery(buffers_[idx].event()) == cudaSuccess)
        {
            buffers_[idx].mark_free();
            inflight_--;
            write_idx_ = (idx + 1) % N;
            return buffers_[idx];
        }
    }

    // 2. 全部 busy，等待最早完成的那个（通常是 read_idx_）
    auto &buf = buffers_[read_idx_];

    CUDA_CHECK(cudaEventSynchronize(buf.event()));
    buf.mark_free();
    inflight_--;

    write_idx_ = (read_idx_ + 1) % N;
    return buf;
}

template <typename T, int N>
void triple_pipeline<T, N>::submit(stream_buffer<T> &buf)
{
    CUDA_CHECK(cudaEventRecord(buf.event(), buf.stream()));
    buf.mark_busy();
    inflight_++;
}
template <typename T, int N>
stream_buffer<T> *triple_pipeline<T, N>::try_fetch()
{
    for (int i = 0; i < N; ++i)
    {
        auto &buf = buffers_[i];
        if (!buf.busy())
            continue;

        if (cudaEventQuery(buf.event()) == cudaSuccess)
        {
            buf.mark_free();
            inflight_--;
            read_idx_ = (idx + 1) % N;
            return &buf;
        }
    }

    return nullptr;
}

template <typename T, int N>
void triple_pipeline<T, N>::release(stream_buffer<T> &buf)
{
    assert(buf.busy());
    buf.mark_free();
    inflight_--;
}
/**
 * @brief
 *
 *
 *
 *
 pipeline<float> pipe(W * H);
 auto& buf = pipeline.acquire();

std::memcpy(buf.h_in(), input, bytes);

cudaMemcpyAsync(buf.d_in(), buf.h_in(), bytes,
                cudaMemcpyHostToDevice, buf.stream());

launch_kernel<<<grid, block, 0, buf.stream()>>>(
    buf.d_in(), buf.d_out());

cudaMemcpyAsync(buf.h_out(), buf.d_out(), bytes,
                cudaMemcpyDeviceToHost, buf.stream());

pipeline.submit(buf);
拉取结果（非阻塞）
if (auto* done = pipeline.try_fetch()) {
    consume(done->h_out());
}
 *
 */