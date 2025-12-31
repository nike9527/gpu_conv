#include <gtest/gtest.h>
#include "test_utils.hpp"
#include "cpu_reference.hpp"

void gaussian_pipeline_run(
    const float* in,
    float* out,
    int w,
    int h
);

TEST(Integration, MixedSizes) {
    std::vector<std::pair<int,int>> sizes = {
        {31,17},
        {128,72},
        {63,63},
        {320,240}
    };

    for (auto [w,h] : sizes) {
        int n = w * h;
        std::vector<float> in(n), out(n), ref(n);

        for (int i = 0; i < n; ++i)
            in[i] = std::sin(i * 0.02f);

        cpu_gaussian(in.data(), ref.data(), w, h);
        gaussian_pipeline_run(in.data(), out.data(), w, h);

        expect_image_near(ref.data(), out.data(), n);
    }
}
                                     