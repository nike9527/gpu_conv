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

}
