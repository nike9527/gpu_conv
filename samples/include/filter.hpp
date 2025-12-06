#pragma once
#include "image.hpp"
#include "kernel.hpp"
#include <string>
#include "image.hpp"

namespace gconv {

enum class Backend {
    CPU_OMP,
    GPU_GLOBAL
};

bool convolve(const Image& src, Image& dst, const Kernel& k, Backend backend);

/**
 * @brief 自定义卷积函数
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param k 内核矩阵
 * @param backend 
 * @return true 
 * @return false 
 */
bool filter(const std::string& src, const std::string& dest, const Kernel& k, Backend backend = Backend::CPU_OMP);
/**
 * @brief 高斯滤波入口（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool gaussianFilter();
/**
 * @brief Sobel 边缘检测（水平）（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool sobelXFilter(const std::string& src, const std::string& dest,Backend backend = Backend::CPU_OMP);
/**
 * @brief Sobel 边缘检测（垂直）（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool sobelYFilter(const std::string& src, const std::string& dest,Backend backend = Backend::CPU_OMP);
/**
 * @brief 锐化滤波器（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool sharpenFilter(const std::string& src, const std::string& dest,Backend backend = Backend::CPU_OMP);
/**
 * @brief 均值模糊滤波器（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool meanBlurFilter(const std::string& src, const std::string& dest,Backend backend = Backend::CPU_OMP);
/**
 * @brief 拉普拉斯算子（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool laplacianFilter(const std::string& src, const std::string& dest,Backend backend = Backend::CPU_OMP);
}
