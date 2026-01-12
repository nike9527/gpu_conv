#pragma once
#include "cuda/cuda_memory.hpp"
#include "cuda/cuda_stream.hpp"
#include <vector>
// template <typename T = float>
class filter_pipeline
{
public:
    cuda_stream stream;         // 专属 stream
    cuda_memory<float> d_input; // 持久 device buffer
    cuda_memory<float> d_output;

    int width;
    int height;
    /**
     * @brief Construct a new filter pipeline object
     *
     * @param w
     * @param h
     * @param flags cudaStreamDefault 同步 cudaStreamNonBlocking (非阻塞)
     */
    filter_pipeline(const int w, const int h, unsigned int flags = cudaStreamDefault)
        : stream(flags), d_input(w * h), d_output(w * h), width(w), height(h)
    {
    }
    ~filter_pipeline() {}
};
