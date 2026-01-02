#include <gtest/gtest.h>
#include <chrono>
#include "convolution_gpu.hpp"
#include "convolution_cpu.hpp"
#include "kernel.hpp"

TEST(Gaussian, Basic_32x32) {
    int w = 32, h = 32, n = w * h;
    std::vector<float> in(n), cpu(n), gpu(n);
    for (int i = 0; i < n; ++i)
        in[i] = i % 13;
    gaussianConvolution(in.data(), cpu.data(), w, h, 3, 0.5);
    gaussianConvolutionGPU(in.data(), gpu.data(), w, h, 3, 0.5);
    expect_image_near(cpu.data(), gpu.data(), n);
}
