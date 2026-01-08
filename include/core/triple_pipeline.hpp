#pragma once
#include "stream_buffer.hpp"
#include "cuda/cuda_memory.hpp"
template<typename T, int N = 3>
class triple_pipeline
{
public:
    explicit triple_pipeline(size_t max_elements)
    {
        for (auto& buf : buffers_)
            buf.allocate(max_elements);
    }
    // 获取一个可写 buffer（可能等待,只返回可写 buffer）
    stream_buffer<T>& acquire();
    // 提交 GPU 任务后调用
    void submit(stream_buffer<T>& buf);
    // 非阻塞取回一个完成的 buffer(只读取已完成 buffer)
    stream_buffer<T>* try_fetch();
private:
    stream_buffer<T> buffers_[N];
    int write_idx_ = 0;
    int read_idx_  = 0;
    int inflight_  = 0;
};

template<typename T, int N>
stream_buffer<T>& triple_pipeline<T,N>::acquire() {
    auto& buf = buffers_[write_idx_];

    if (buf.busy()) {
        CUDA_CHECK(cudaEventSynchronize(buf.event()));
        buf.mark_free();
        inflight_--;
    }

    write_idx_ = (write_idx_ + 1) % N;
    return buf;
}
template<typename T, int N>
void triple_pipeline<T,N>::submit(stream_buffer<T>& buf){
    CUDA_CHECK(cudaEventRecord(buf.event(), buf.stream()));
    buf.mark_busy();
    inflight_++;
}
template<typename T, int N>
stream_buffer<T>* triple_pipeline<T, N>::try_fetch() {
    auto& buf = buffers_[read_idx_];

    if (!buf.busy())
        return nullptr;

    if (cudaEventQuery(buf.event()) != cudaSuccess)
        return nullptr;

    buf.mark_free();
    inflight_--;
    read_idx_ = (read_idx_ + 1) % N;
    return &buf;
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