#pragma once
#include "bench_types.hpp"
#include <vector>
struct BenchCase {
    int width;
    int height;
    int kSize;
    int block_x;
    int block_y;
    int block_z;
    int grid_x;
    int grid_y;
    int grid_z;
    filter_type filter;
    mem_type mType;
    PipelineType pipeline;
};
struct BenchResult {float kernel_ms;float gpixel;};
/**
 * @brief  高斯测试用例
 * 
 * @return std::vector<BenchCase> 
 */
// inline std::vector<BenchCase> getBenchCases() {
//     return {
//         {1920, 1080, 3, 0, 0, 0, 0, 0, 0, filter_type::SOBEL,    mem_type::GLOBAL,       PipelineType::SINGLE_STREAM},
//         {1920, 1080, 3, 0, 0, 0, 0, 0, 0, filter_type::SOBEL,    mem_type::SHAREDCONST, PipelineType::SINGLE_STREAM},
//         {1920, 1080, 5, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::GLOBAL,       PipelineType::TRIPLE_BUFFER},
//         {1920, 1080, 5, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::SHAREDCONST, PipelineType::TRIPLE_BUFFER},
//         {3840, 2160, 5, 0, 0, 0, 0, 0, 0, filter_type::MEAN,     mem_type::SHAREDCONST, PipelineType::TRIPLE_BUFFER},
//     };
// }
// inline std::vector<BenchCase> getBenchCases() {
//     return {
//         {2560, 1440, 3, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::GLOBAL,       PipelineType::SINGLE_STREAM},
//         {2560, 1440, 3, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::SHAREDCONST,  PipelineType::SINGLE_STREAM},

//         {2560, 1440, 5, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::GLOBAL,       PipelineType::SINGLE_STREAM},
//         {2560, 1440, 5, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::SHAREDCONST,  PipelineType::SINGLE_STREAM},

//         {2560, 1440, 7, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::GLOBAL,       PipelineType::SINGLE_STREAM},
//         {2560, 1440, 7, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::SHAREDCONST,  PipelineType::SINGLE_STREAM},

//         {2560, 1440, 9, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::GLOBAL,       PipelineType::SINGLE_STREAM},
//         {2560, 1440, 9, 0, 0, 0, 0, 0, 0, filter_type::GAUSSIAN, mem_type::SHAREDCONST,  PipelineType::SINGLE_STREAM},
//     };
// }

inline std::vector<BenchCase> getBenchCases() {
    return {
        {2560, 1440, 3, 0, 0, 0, 0, 0, 0, filter_type::SHARPEN, mem_type::GLOBAL,       PipelineType::SINGLE_STREAM},
        {2560, 1440, 3, 0, 0, 0, 0, 0, 0, filter_type::SHARPEN, mem_type::SHAREDCONST,  PipelineType::SINGLE_STREAM},

        {2560, 1440, 5, 0, 0, 0, 0, 0, 0, filter_type::SHARPEN, mem_type::GLOBAL,       PipelineType::SINGLE_STREAM},
        {2560, 1440, 5, 0, 0, 0, 0, 0, 0, filter_type::SHARPEN, mem_type::SHAREDCONST,  PipelineType::SINGLE_STREAM},

        {2560, 1440, 7, 0, 0, 0, 0, 0, 0, filter_type::SHARPEN, mem_type::GLOBAL,       PipelineType::TRIPLE_BUFFER},
        {2560, 1440, 7, 0, 0, 0, 0, 0, 0, filter_type::SHARPEN, mem_type::SHAREDCONST,  PipelineType::TRIPLE_BUFFER},

        {2560, 1440, 9, 0, 0, 0, 0, 0, 0, filter_type::SHARPEN, mem_type::GLOBAL,       PipelineType::TRIPLE_BUFFER},
        {2560, 1440, 9, 0, 0, 0, 0, 0, 0, filter_type::SHARPEN, mem_type::SHAREDCONST,  PipelineType::TRIPLE_BUFFER},
    };
}