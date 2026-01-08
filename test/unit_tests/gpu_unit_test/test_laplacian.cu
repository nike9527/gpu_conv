#include <gtest/gtest.h>
#include <chrono>
#include "convolution_gpu.hpp"
#include "convolution_cpu.hpp"
#include "filter.hpp"


TEST(Laplacian, Edge) {
    int w = 16, h = 16, n = w * h;
    std::vector<float> in(n), cpu(n), gpu(n);

    for (int i = 0; i < n; ++i)
        in[i] = (i % w < w/2) ? 0.f : 1.f;
    laplacianConvolution(in.data(), cpu.data(), w, h, lap, 3);

    laplacianConvolutionGPU(in.data(), gpu.data(), w, h);

    expect_image_near(cpu.data(), gpu.data(), n);
}
