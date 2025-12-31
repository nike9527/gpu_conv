#pragma once
#include "cuda_check.hpp"

class cuda_stream {
public:
    cuda_stream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~cuda_stream() noexcept {
        if (stream_) cudaStreamDestroy(stream_);
    }
    // 禁止拷贝
    cuda_stream(const cuda_stream&) = delete;
    cuda_stream& operator=(const cuda_stream&) = delete;

    operator cudaStream_t() const noexcept { return stream_; }

    cuda_stream(cuda_stream&& other) noexcept {
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    cuda_stream& operator=(cuda_stream&& other){
        stream_ = other.stream_;
        other.stream_ = nullptr;
        return *this;
    }

    cudaStream_t get() const noexcept {return stream_;}

    void sync() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bool query() const {
        if (!stream_) return true;
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        throw cuda_error::cuda_exception(err, __FILE__, __LINE__);
    }
    // 等待事件
    void wait_event(const cudaEvent_t& event) {
        CUDA_CHECK(cudaStreamWaitEvent(stream_, event, 0));
    }
private:
    cudaStream_t stream_ = nullptr;
};
