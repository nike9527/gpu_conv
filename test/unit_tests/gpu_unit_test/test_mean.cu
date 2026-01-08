#include <gtest/gtest.h>
#include <chrono>
#include "convolution_gpu.hpp"
#include "convolution_cpu.hpp"
#include "filter.hpp"

TEST(Mean, NonAligned_31x17) {
    int w = 31, h = 17, n = w * h;
    std::vector<float> in(n), cpu(n), gpu(n);

    for (int i = 0; i < n; ++i)
        in[i] = i * 0.01f;

    meanBlurConvolution(in.data(), cpu.data(), w, h, 3);
    meanBlurConvolutionGPU(in.data(), gpu.data(), w, h, 3);

    expect_image_near(cpu.data(), gpu.data(), n);
}
