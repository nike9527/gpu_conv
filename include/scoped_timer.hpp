#pragma once
#include "cuda_event.hpp"
#include <string>
#include <iostream>

class scoped_cuda_timer {
public:
    scoped_cuda_timer(const std::string& name, cudaStream_t stream = 0)
        : name_(name), stream_(stream)
    {
        start_.record(stream_);
    }

    ~scoped_cuda_timer() {
        stop_.record(stream_);
        stop_.sync();
        float ms = start_.elapsed_ms(stop_);
        std::cout << "[CUDA TIMER] " << name_ << " : " << ms << " ms\n";
    }

private:
    std::string name_;
    cudaStream_t stream_;
    cuda_event start_, stop_;
};
