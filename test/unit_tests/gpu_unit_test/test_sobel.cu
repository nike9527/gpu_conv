#include <gtest/gtest.h>
#include <chrono>
#include "convolution_gpu.hpp"
#include "convolution_cpu.hpp"
#include "filter.hpp"


void run_sobel_cuda(const float*, float*, int, int);

TEST(Sobel, Magnitude) {
    int w = 64, h = 64, n = w * h;
    std::vector<float> in(n), gx(n), gy(n), cpu(n), gpu(n);

    for (int i = 0; i < n; ++i)
        in[i] = std::sin(i * 0.05f);

    sobelConvolutionX(in.data(), gx.data(), w, h, sx, 3);
    sobelConvolutionY(in.data(), gy.data(), w, h, sy, 3);

    for (int i = 0; i < n; ++i)
        cpu[i] = std::sqrt(gx[i]*gx[i] + gy[i]*gy[i]);

    sobelConvolutionGPU(in.data(), gpu.data(), w, h);

    expect_image_near(cpu.data(), gpu.data(), n, 1e-3f);
}
