#pragma once
#include "cuda_exception.hpp"
#include <iostream>
class cuda_stream
{
public:
    /**
     * @brief 默认构造 默认流 (stream = 0)
     */
    cuda_stream() noexcept : stream_(0), owning_(false) {}
    cuda_stream(unsigned int flags) : owning_(true)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    ~cuda_stream() noexcept
    {
        release();
    }
    // 禁止拷贝
    cuda_stream(const cuda_stream &) = delete;
    cuda_stream &operator=(const cuda_stream &) = delete;

    operator cudaStream_t() const noexcept { return stream_; }

    cuda_stream(cuda_stream &&other) noexcept
    {
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    cuda_stream &operator=(cuda_stream &&other)
    {
        stream_ = other.stream_;
        other.stream_ = nullptr;
        return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }
    bool is_default() const noexcept
    {
        return stream_ == 0;
    }
    void synchronize() const
    {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bool query() const
    {
        if (!stream_)
            return true;
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess)
            return true;
        if (err == cudaErrorNotReady)
            return false;
        throw cuda_error::cuda_exception(err, __FILE__, __LINE__);
    }
    // 等待事件
    void wait_event(const cudaEvent_t &event)
    {
        CUDA_CHECK(cudaStreamWaitEvent(stream_, event, 0));
    }
    void release() noexcept
    {
        if (owning_ && stream_)
        {
            cudaStreamDestroy(stream_);
        }
        stream_ = nullptr;
        owning_ = false;
    }

private:
    // 是否使用默认流 true 默认流 false 创建流
    bool owning_;
    cudaStream_t stream_ = 0;
};
