#pragma once
#include "filters/filter.hpp"
#include "core/filter_pipeline.hpp"
namespace gpu_conv
{
    /**
     * @brief 自自定义卷积
     *
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param kernel 内核
     * @param kSize  核大小
     */
    void conv2dKernel(const float *in, float *out, const int width, const int height, mem_type type, const filter &filterObj);

    /**
     * @brief 高斯卷积(cpu omp)
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param kSize  核大小
     */
    void gaussianBlur(const float *in, float *out, const int width, const int height, mem_type type, int size, float sigma);
    /**
     * @brief 拉普拉斯算子(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     */
    void laplacian(const float *in, float *out, const int width, const int height, mem_type type);
    /**
     * @brief  均值模糊(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     * @param kSize  核大小
     */
    void meanBlur(const float *in, float *out, const int width, const int height, mem_type type, int size);
    /**
     * @brief  锐化滤波器(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     */
    void sharpen(const float *in, float *out, const int width, const int height, mem_type type);
    /**
     * @brief sobel卷积(cpu omp)
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param dx x方向卷积
     * @param dy y方向卷积
     */
    void sobel(const float *in, float *out, const int width, const int height, mem_type type);
    /**
     * @brief 异步 pipeline
     *
     * @param pipe
     * @param in
     * @param out
     * @param width
     * @param height
     * @param type
     * @param block_w
     * @param block_h
     */
    void launchFilterAsync(filter_pipeline &pipe, const float *in, float *out, const int width, const int height, const filter_type type, int block_w = 16, int block_h = 16);
};