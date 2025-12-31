#include <gtest/gtest.h>
#include "cpu_reference.hpp"
#include "test_utils.hpp"

void run_laplacian_cuda(const float*, float*, int, int);

TEST(Laplacian, Edge) {
    int w = 16, h = 16, n = w * h;
    std::vector<float> in(n), cpu(n), gpu(n);

    for (int i = 0; i < n; ++i)
        in[i] = (i % w < w/2) ? 0.f : 1.f;

    std::vector<float> lap = {
         0, 1, 0,
         1,-4, 1,
         0, 1, 0
    };

    cpu_convolution(in.data(), cpu.data(), w, h, lap, 3);
    run_laplacian_cuda(in.data(), gpu.data(), w, h);

    expect_image_near(cpu.data(), gpu.data(), n);
}
