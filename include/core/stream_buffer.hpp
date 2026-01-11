#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda/cuda_stream.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_memory.hpp"
enum class buffer_state : uint8_t
{
    FREE,     // 可 acquire：GPU 不使用 == pipeline: 不占用 不持有 ==  CPU: 不可读
    INFLIGHT, // 已 submit：GPU 可能正在使用  == pipeline: 持有  == CPU: 不可读
    COMPLETED // GPU: 已完成，不使用 == pipeline: 仍持有（等待 CPU 消费） == CPU:可读
};
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

    ~stream_buffer() { release_async(); }

    void allocate(size_t elements)
    {
        if (elements <= capacity_)
            return;

        // release();
        release_async();

        capacity_ = elements;
        size_t bytes = elements * sizeof(T);

        CUDA_CHECK(cudaMallocHost(&h_in_, bytes));
        CUDA_CHECK(cudaMallocHost(&h_out_, bytes));

        d_in_ = cuda_memory<T>(elements);
        d_out_ = cuda_memory<T>(elements);
    }

    void release() noexcept
    {
        // if (busy_)
        //     cudaEventSynchronize(event_.get());
        if (h_in_)
            cudaFreeHost(h_in_);
        if (h_out_)
            cudaFreeHost(h_out_);
        h_in_ = h_out_ = nullptr;
        // busy_ = false;
        capacity_ = 0;
        state_ = buffer_state::FREE;
    }
    // ===== 状态查询 =====
    buffer_state state() const noexcept { return state_; }
    bool is_free() const noexcept { return state_ == buffer_state::FREE; }
    bool is_inflight() const noexcept { return state_ == buffer_state::INFLIGHT; }
    bool is_completed() const noexcept { return state_ == buffer_state::COMPLETED; }
    // ===== 状态转移（只允许 pipeline 调用）=====
    void mark_inflight() noexcept
    {
        state_ = buffer_state::INFLIGHT;
    }
    void mark_completed() noexcept
    {
        state_ = buffer_state::COMPLETED;
    }

    void mark_free() noexcept
    {
        state_ = buffer_state::FREE;
    }

    void release_async() noexcept
    {
        // 异步释放：先查询 event，如果还在 GPU 执行，不阻塞
        if (busy_ && cudaEventQuery(event_.get()) != cudaSuccess)
        {
            // GPU 还在使用 → buffer 保留
            return;
        }

        if (h_in_)
            cudaFreeHost(h_in_);
        if (h_out_)
            cudaFreeHost(h_out_);
        h_in_ = h_out_ = nullptr;
        capacity_ = 0;
        busy_ = false;
    }
    T *h_in() noexcept { return h_in_; }
    T *h_out() noexcept { return h_out_; }

    T *d_in() noexcept { return d_in_.data(); }
    T *d_out() noexcept { return d_out_.data(); }

    cudaStream_t stream() const noexcept { return stream_.get(); }
    cudaEvent_t event() const noexcept { return event_.get(); }

    size_t capacity() const noexcept { return capacity_; }
    // bool busy() const noexcept { return busy_; }
    // void mark_busy() noexcept { busy_ = true; }
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
        state_ = other.state_;

        other.h_in_ = other.h_out_ = nullptr;
        other.capacity_ = 0;
        other.state_ = buffer_state::FREE;
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
    // bool busy_ = false;
    buffer_state state_ = buffer_state::FREE;
};
