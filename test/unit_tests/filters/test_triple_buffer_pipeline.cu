#include <gtest/gtest.h>
#include "test_utils.hpp"
#include "cpu_reference.hpp"
#include "triple_buffer_pipeline.hpp"

void run_mean_pipeline(
    triple_buffer_pipeline& pipe,
    const float* in,
    float* out,
    int w,
    int h
);

TEST(Pipeline, TripleBuffer_MixedSizes) {
    triple_buffer_pipeline pipe(1920 * 1080);

    std::vector<std::pair<int,int>> sizes = {
        {64,64}, {128,72}, {31,17}, {320,240}
    };

    std::vector<float> mean(9, 1.f/9.f);

    for (auto [w,h] : sizes) {
        int n = w*h;
        std::vector<float> in(n), cpu(n), out(n);

        for (int i = 0; i < n; ++i)
            in[i] = i * 0.02f;

        cpu_convolution(in.data(), cpu.data(), w, h, mean, 3);
        run_mean_pipeline(pipe, in.data(), out.data(), w, h);

        expect_image_near(cpu.data(), out.data(), n);
    }
}
