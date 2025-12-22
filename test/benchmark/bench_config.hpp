#pragma once
#include "bench_types.hpp"
#include <vector>

struct BenchCase {
    int width;
    int height;
    int kSize;
    FilterType filter;
    MemType mType;
    PipelineType pipeline;
};
struct BenchResult {
    float kernel_ms;
    float gpixel;
};
/**
 * @brief  测试用例
 * 
 * @return std::vector<BenchCase> 
 */
inline std::vector<BenchCase> getBenchCases() {
    return {
        {1920, 1080, 3, FilterType::SOBEL,    MemType::GLOBAL,       PipelineType::SINGLE_STREAM},
        {1920, 1080, 3, FilterType::SOBEL,    MemType::SHARED_CONST, PipelineType::SINGLE_STREAM},
        {1920, 1080, 5, FilterType::GAUSSIAN, MemType::GLOBAL,       PipelineType::TRIPLE_BUFFER},
        {1920, 1080, 5, FilterType::GAUSSIAN, MemType::SHARED_CONST, PipelineType::TRIPLE_BUFFER},
        {3840, 2160, 5, FilterType::MEAN,     MemType::SHARED_CONST, PipelineType::TRIPLE_BUFFER},
    };
}
