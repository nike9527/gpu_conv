#pragma once
#include <string>
#include "kernel.hpp"
/**
 * @brief 内存的类型
 */
enum class MemType {
    GLOBAL,             //全局内存
    SHARED_CONST        //共享内存+常量内存
};
/**
 * @brief  过滤器
 */
enum class FilterType {
    SOBEL,              
    GAUSSIAN,
    MEAN,
    SHARPEN,
    LAPLACIAN,
    CUSTOM
};

enum class PipelineType {
    SINGLE_STREAM,
    TRIPLE_BUFFER
};

inline const char* toString(MemType k) {
    return k == MemType::GLOBAL ? "Global" : "Shared+Const";
}

inline const Kernel getKernel(FilterType f, int ksize = 3) {
    switch (f) {
        case FilterType::SOBEL:     return Kernel::sobelX();
        case FilterType::GAUSSIAN:  return Kernel::gaussian(ksize,3.0);
        case FilterType::MEAN:      return Kernel::meanBlur(ksize);
        case FilterType::SHARPEN:   return Kernel::sharpen();
        case FilterType::LAPLACIAN: return Kernel::laplacian();
        default:                    return Kernel::filterKernel(3,{1,1,1,1,1,1,1,1,1});
    }
}

inline const char* toString(FilterType f) {
    switch (f) {
        case FilterType::SOBEL:     return "Sobel";
        case FilterType::GAUSSIAN:  return "Gaussian";
        case FilterType::MEAN:      return "Mean";
        case FilterType::SHARPEN:   return "Sharpen";
        case FilterType::LAPLACIAN: return "Laplacian";
        default:                    return "Custom";
    }
}

inline const char* toString(PipelineType p) {
    return p == PipelineType::SINGLE_STREAM ? "Single" : "Triple";
}
