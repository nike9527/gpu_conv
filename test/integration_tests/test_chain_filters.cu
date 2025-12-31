#include <gtest/gtest.h>
#include "test_utils.hpp"
#include "cpu_reference.hpp"

void pipeline_gaussian_sobel_mean(
    const float* in,
    float* out,
    int w,
    int h
);

TEST(Integration, Gaussian_Sobel_Mean) {
    int w = 64, h = 64, n = w * h;

    std::vector<float> in(n), out(n);
    std::vector<float> t1(n), t2(n), ref(n);

    for (int i = 0; i < n; ++i)
        in[i] = (i % 19) * 0.07f;

    // CPU chain
    cpu_gaussian(in.data(), t1.data(), w, h);
    cpu_sobel(t1.data(), t2.data(), w, h);
    cpu_mean(t2.data(), ref.data(), w, h);

    // GPU chain
    pipeline_gaussian_sobel_mean(in.data(), out.data(), w, h);

    expect_image_near(ref.data(), out.data(), n, 1e-3f);
}
