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

TEST(Integration, TripleBuffer_Stress) {
    triple_buffer_pipeline pipe(1920 * 1080);

    for (int frame = 0; frame < 30; ++frame) {
        int w = (frame % 2) ? 128 : 256;
        int h = (frame % 3) ? 72  : 144;
        int n = w * h;

        std::vector<float> in(n), out(n), ref(n);

        for (int i = 0; i < n; ++i)
            in[i] = frame + i * 0.001f;

        cpu_mean(in.data(), ref.data(), w, h);
        run_mean_pipeline(pipe, in.data(), out.data(), w, h);

        expect_image_near(ref.data(), out.data(), n);
    }
}
