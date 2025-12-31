#include <gtest/gtest.h>
#include "test_utils.hpp"
#include "cpu_reference.hpp"

// 你工程里的真实接口
void gaussian_pipeline_run(
    const float* in,
    float* out,
    int w,
    int h
);

TEST(Integration, Gaussian_E2E) {
    int w = 128, h = 128, n = w * h;

    std::vector<float> in(n), out(n), ref(n);

    for (int i = 0; i < n; ++i)
        in[i] = std::cos(i * 0.01f);

    // CPU reference
    std::vector<float> gaussian = {
        1,2,1,
        2,4,2,
        1,2,1
    };
    for (auto& v : gaussian) v /= 16.f;

    cpu_convolution(in.data(), ref.data(), w, h, gaussian, 3);

    // GPU pipeline
    gaussian_pipeline_run(in.data(), out.data(), w, h);

    expect_image_near(ref.data(), out.data(), n);
}
