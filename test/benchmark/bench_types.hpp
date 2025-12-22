#pragma once
#include <string>
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
