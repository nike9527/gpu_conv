#pragma once
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_stream.hpp"
#include <cstdio>
#include <cuda_runtime.h>
class gpu_timer {
private:
    cuda_event start, stop;
public:
    gpu_timer() = default;
    ~gpu_timer() noexcept = default;
    gpu_timer(const gpu_timer&) = delete;
    gpu_timer& operator=(const gpu_timer&) = delete;
    gpu_timer(gpu_timer&& other)  = delete;
    gpu_timer& operator=(gpu_timer&& other)  = delete;
    void tic(cuda_stream& stream) {
        start.record(stream.get());
    }

    float toc(cuda_stream& stream) {
        stop.record(stream.get());
        stop.synchronize();
        return start.elapsed_ms(stop);
    }
};
