#pragma once
#include "bench_config.hpp"

struct BenchResult
{
    double kernel_ms;
    double h2d_ms;
    double d2h_ms;
};

BenchResult runGaussianOnce(
    int width,
    int height,
    int ksize,
    mem_type mem,
    PipelineType pipeline // Single / Triple
);
