#include <gtest/gtest.h>
#include "cpu_reference.hpp"
#include "test_utils.hpp"

// 你已有的 CUDA API
void run_gaussian_cuda(const float* in,float* out,int w,int h);

TEST(Gaussian, Basic_32x32) {
    int w = 32, h = 32, n = w * h;

    std::vector<float> in(n), cpu(n), gpu(n);

    for (int i = 0; i < n; ++i)
        in[i] = i % 13;

    std::vector<float> gaussian = {
        1,2,1,
        2,4,2,
        1,2,1
    };
    for (auto& v : gaussian) v /= 16.f;

    cpu_convolution(in.data(), cpu.data(), w, h, gaussian, 3);
    run_gaussian_cuda(in.data(), gpu.data(), w, h);

    expect_image_near(cpu.data(), gpu.data(), n);
}
