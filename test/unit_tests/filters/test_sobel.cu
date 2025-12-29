#include <gtest/gtest.h>
#include <cmath>
#include "cpu_reference.hpp"
#include "test_utils.hpp"

void run_sobel_cuda(const float*, float*, int, int);

TEST(Sobel, Magnitude) {
    int w = 64, h = 64, n = w * h;
    std::vector<float> in(n), gx(n), gy(n), cpu(n), gpu(n);

    for (int i = 0; i < n; ++i)
        in[i] = std::sin(i * 0.05f);

    std::vector<float> sx = {
        -1,0,1,
        -2,0,2,
        -1,0,1
    };
    std::vector<float> sy = {
        -1,-2,-1,
         0, 0, 0,
         1, 2, 1
    };

    cpu_convolution(in.data(), gx.data(), w, h, sx, 3);
    cpu_convolution(in.data(), gy.data(), w, h, sy, 3);

    for (int i = 0; i < n; ++i)
        cpu[i] = std::sqrt(gx[i]*gx[i] + gy[i]*gy[i]);

    run_sobel_cuda(in.data(), gpu.data(), w, h);

    expect_image_near(cpu.data(), gpu.data(), n, 1e-3f);
}
