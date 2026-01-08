#include "bench_types.hpp"
#include "bench_config.hpp"
#include <iostream>

void printResult(const BenchCase& c, float ms, float gpixel) {
    std::cout
        << "[Benchmark]\n"
        << "Filter:   " << filter::getFilterName(c.filter) << "\n"
        << "memory:   " << toString(c.mType) << "\n"
        << "Pipeline: " << toString(c.pipeline) << "\n"
        << "Res:      " << c.width << "x" << c.height << "\n"
        << "kSize:    " << c.kSize << "\n"
        << "block:    (" << c.block_x<<", "<<c.block_y<<", "<<c.block_z<<")\n"
        << "grid:     (" << c.grid_x<<", "<<c.grid_y<<", "<<c.grid_z<<")\n"
        << "Kernel:   " << ms << " ms\n"
        << "Throughput: " << gpixel << " GPixel/s\n\n";
}
