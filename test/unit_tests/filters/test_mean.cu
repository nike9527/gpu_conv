#include <gtest/gtest.h>
#include "cpu_reference.hpp"
#include "test_utils.hpp"

void run_mean_cuda(const float*, float*, int, int);

TEST(Mean, NonAligned_31x17) {
    int w = 31, h = 17, n = w * h;
    std::vector<float> in(n), cpu(n), gpu(n);

    for (int i = 0; i < n; ++i)
        in[i] = i * 0.01f;

    std::vector<float> mean(9, 1.f / 9.f);

    cpu_convolution(in.data(), cpu.data(), w, h, mean, 3);
    run_mean_cuda(in.data(), gpu.data(), w, h);

    expect_image_near(cpu.data(), gpu.data(), n);
}
