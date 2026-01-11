#pragma once
#include <cuda_runtime.h>
#include "buffer_state.hpp"
#include "cuda/cuda_stream.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_memory.hpp"
/**
 * @brief 资源 + 状态
 * stream_buffer 现在是 FramePipeline 的“私有实现细节
 * @tparam T
 */
template <typename T>
class stream_buffer
{
public:
    explicit stream_buffer(size_t capacity)
        : capacity_(capacity),
          d_in_(capacity),
          d_out_(capacity)
    {
        size_t bytes = capacity * sizeof(T);
        CUDA_CHECK(cudaMallocHost(&h_in_, bytes));
        CUDA_CHECK(cudaMallocHost(&h_out_, bytes));
    }

    ~stream_buffer()
    {
        if (h_in_)
            cudaFreeHost(h_in_);
        if (h_out_)
            cudaFreeHost(h_out_);
    }

    stream_buffer(const stream_buffer &) = delete;
    stream_buffer &operator=(const stream_buffer &) = delete;

    // ===== 状态 =====
    buffer_state state() const noexcept { return state_; }

    void mark_inflight() noexcept { state_ = buffer_state::INFLIGHT; }
    void mark_completed() noexcept { state_ = buffer_state::COMPLETED; }
    void mark_free() noexcept { state_ = buffer_state::FREE; }

    // ===== 资源访问 =====
    T *h_in() noexcept { return h_in_; }
    T *h_out() noexcept { return h_out_; }
    T *d_in() noexcept { return d_in_.data(); }
    T *d_out() noexcept { return d_out_.data(); }

    cudaStream_t stream() const noexcept { return stream_.get(); }
    cudaEvent_t event() const noexcept { return event_.get(); }

private:
    T *h_in_{nullptr};
    T *h_out_{nullptr};

    size_t capacity_{0};
    cuda_memory<T> d_in_;
    cuda_memory<T> d_out_;

    cuda_stream stream_;
    cuda_event event_;
    buffer_state state_{buffer_state::FREE};
};
