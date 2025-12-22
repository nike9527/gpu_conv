#include "bench_types.hpp"
#include "bench_config.hpp"
#include <iostream>

void printResult(const BenchCase& c, float ms, float gpixel) {
    std::cout
        << "[Benchmark]\n"
        << "Filter:   " << toString(c.filter) << "\n"
        << "memory:   " << toString(c.mType) << "\n"
        << "Pipeline: " << toString(c.pipeline) << "\n"
        << "Res:      " << c.width << "x" << c.height << "\n"
        << "kSize:    " << c.kSize << "\n"
        << "Kernel:   " << ms << " ms\n"
        << "Throughput: " << gpixel << " GPixel/s\n\n";
}
