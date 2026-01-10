#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda/cuda_stream.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_memory.hpp"
inline static std::atomic<int> next_id_{0};
template <typename T>
class stream_buffer
{
public:
    stream_buffer() : id_(next_id_++) {};
    explicit stream_buffer(size_t max_elements) { allocate(max_elements); }
    // 禁止拷贝
    stream_buffer(const stream_buffer &) = delete;
    stream_buffer &operator=(const stream_buffer &) = delete;
    // 允许移动
    stream_buffer(stream_buffer &&other) noexcept { move_from(std::move(other)); }
    stream_buffer &operator=(stream_buffer &&other) noexcept
    {
        if (this != &other)
        {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    ~stream_buffer() { release(); }

    void allocate(size_t elements)
    {
        if (elements <= capacity_)
            return;

        release();

        capacity_ = elements;
        size_t bytes = elements * sizeof(T);

        CUDA_CHECK(cudaMallocHost(&h_in_, bytes));
        CUDA_CHECK(cudaMallocHost(&h_out_, bytes));

        d_in_ = cuda_memory<T>(elements);
        d_out_ = cuda_memory<T>(elements);
    }

    void release() noexcept
    {
        if (busy_)
            cudaEventSynchronize(event_.get());
        if (h_in_)
            cudaFreeHost(h_in_);
        if (h_out_)
            cudaFreeHost(h_out_);
        h_in_ = h_out_ = nullptr;
        busy_ = false;
        capacity_ = 0;
    }

    T *h_in() noexcept { return h_in_; }
    T *h_out() noexcept { return h_out_; }

    T *d_in() noexcept { return d_in_.data(); }
    T *d_out() noexcept { return d_out_.data(); }

    cudaStream_t stream() const noexcept { return stream_.get(); }
    cudaEvent_t event() const noexcept { return event_.get(); }

    size_t capacity() const noexcept { return capacity_; }
    bool busy() const noexcept { return busy_; }
    void mark_busy() noexcept { busy_ = true; }
    void mark_free() noexcept { busy_ = false; }
    int id() const noexcept { return id_; }

private:
    void move_from(stream_buffer &&other) noexcept
    {
        h_in_ = other.h_in_;
        h_out_ = other.h_out_;
        capacity_ = other.capacity_;

        d_in_ = std::move(other.d_in_);
        d_out_ = std::move(other.d_out_);
        stream_ = std::move(other.stream_);
        event_ = std::move(other.event_);

        other.h_in_ = other.h_out_ = nullptr;
        other.capacity_ = 0;
    }

private:
    int id_{0};
    T *h_in_ = nullptr;
    T *h_out_ = nullptr;
    cuda_memory<T> d_in_;
    cuda_memory<T> d_out_;
    size_t capacity_ = 0;
    cuda_stream stream_{cudaStreamNonBlocking};
    cuda_event event_;
    bool busy_ = false;
};
