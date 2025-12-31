#pragma once
#include "cuda_check.hpp"

class cuda_event {
public:
    cuda_event(unsigned int flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }
    ~cuda_event() noexcept {
        if (event_) cudaEventDestroy(event_);
    }
    // 禁止拷贝
    cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;
    //可以移动
    cuda_event(cuda_event&& other){
        event_ = other.event_;
        other.event_ = nullptr;
    }
    cuda_event& operator=( cuda_event&& other){
        event_ = other.event_;
        other.event_ = nullptr;
        return *this;
    }

    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }
   void synchronize() const {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    void sync() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    bool query() const {
        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        throw cuda_error::cuda_exception(err, __FILE__, __LINE__);
    }
    float elapsed_ms(const cuda_event& other) const {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms,event_,other.event_));
        return ms;
    }
    // 获取原生句柄
    cudaEvent_t get() const noexcept { return event_; }
    operator cudaEvent_t() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};
