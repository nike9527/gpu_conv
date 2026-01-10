#pragma once
#include "conv/conv_cpu.hpp"
#include <cmath>
#include <omp.h>
namespace cpu_conv
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
    void conv2dKernel(const float *in, float *out, const int w, const int h, const filter &filter)
    {
        int r = filter.size / 2;
#pragma omp parallel for
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float sum = 0.0f;
                for (int ky = -r; ky <= r; ky++)
                {
                    for (int kx = -r; kx <= r; kx++)
                    {
                        int ix = std::min(std::max(x + kx, 0), w - 1);
                        int iy = std::min(std::max(y + ky, 0), h - 1);
                        sum += in[iy * w + ix] * filter.kdata[(ky + r) * filter.size + (kx + r)];
                    }
                }
                out[y * w + x] = sum;
            }
        }
    }
    /**
     * @brief高斯卷积(cpu omp)
     *
     * @param in
     * @param out
     * @param w
     * @param h
     * @param filter
     */
    void gaussianBlur(const float *in, float *out, const int w, const int h, const int size, const float sigma)
    {
        gaussianBlur1D(in, out, w, h, size, sigma);
    }
    /**
     * @brief 高斯卷积(cpu omp) 2D
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param kSize  核大小
     */
    void gaussianBlur2D(const float *in, float *out, const int w, const int h, const int size, const float sigma)
    {
        filter obj = filter::gaussian2D(size, sigma);
        int radius = obj.radius;
        int ksize = obj.size;
#pragma omp parallel for
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    // 使用镜像边界
                    int iy = y + ky;
                    // 镜像处理
                    // if (iy < 0) iy = -iy;
                    // else if (iy >= h) iy = h - 1;
                    if (iy < 0)
                        iy = -iy - 1; // 镜像：0 → -1 → 0, -1 → -2 → 1
                    else if (iy >= h)
                        iy = 2 * h - iy - 1; // 镜像：h → h-1, h+1 → h-2
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        // 使用镜像边界
                        int ix = x + kx;
                        // 镜像处理
                        // if (ix < 0) ix = -ix;
                        // else if (ix >= w) ix = w - 1;
                        if (ix < 0)
                            ix = -ix - 1;
                        else if (ix >= w)
                            ix = 2 * w - ix - 1;
                        sum += in[iy * w + ix] * obj.kdata[(ky + radius) * ksize + (kx + radius)];
                    }
                }
                out[y * w + x] = sum;
            }
        }
    }

    /**
     * @brief 高斯卷积(cpu omp) 1D
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param kSize  核大小
     */
    void gaussianBlur1D(const float *in, float *out, const int w, const int h, const int size, const float sigma)
    {
        filter obj = filter::gaussian1D(size, sigma);
        int r = obj.size / 2;
        std::vector<float> temp(w * h, 0.f);
// 第一步：水平方向卷积
#pragma omp parallel for
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float val = 0.0f;
                for (int k = -r; k <= r; k++)
                {
                    int xk = x + k;
                    // 边界处理：镜像边界
                    if (xk < 0)
                        xk = -xk;
                    if (xk >= w)
                        xk = 2 * w - xk - 1;
                    val += in[y * w + xk] * obj.kdata[k + r];
                }
                temp[y * w + x] = val;
            }
        }

// 第二步：垂直方向卷积
#pragma omp parallel for
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float val = 0.0f;
                for (int k = -r; k <= r; k++)
                {
                    int yk = y + k;
                    // 边界处理：镜像边界
                    if (yk < 0)
                        yk = -yk;
                    if (yk >= h)
                        yk = 2 * h - yk - 1;
                    val += temp[yk * w + x] * obj.kdata[k + r];
                }
                out[y * w + x] = val;
            }
        }
    }

    /**
     * @brief 拉普拉斯算子(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     */
    void laplacian(const float *in, float *out, const int w, const int h)
    {
        filter obj = filter::laplacian();
        int kSize = obj.size;
        int radius = obj.radius;
#pragma omp parallel for
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    // 使用镜像边界
                    int iy = y + ky;
                    if (iy < 0)
                        iy = -iy - 1;
                    else if (iy >= h)
                        iy = 2 * h - iy - 1;
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        // 使用镜像边界
                        int ix = x + kx;
                        if (ix < 0)
                            ix = -ix - 1;
                        else if (ix >= w)
                            ix = 2 * w - ix - 1;
                        sum += in[iy * w + ix] * obj.kdata[(ky + radius) * kSize + (kx + radius)];
                    }
                }
                out[y * w + x] = sum;
            }
        }
    }
    /**
     * @brief  均值模糊(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     * @param kSize  核大小
     */
    void meanBlur(const float *in, float *out, const int w, const int h, int size)
    {
        filter obj = filter::meanBlur(size);

        int ksize = obj.size;
        int radius = obj.radius;
#pragma omp parallel for
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    // 使用镜像边界
                    int iy = y + ky;
                    if (iy < 0)
                        iy = -iy - 1;
                    else if (iy >= h)
                        iy = 2 * h - iy - 1;
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        // 使用镜像边界
                        int ix = x + kx;
                        if (ix < 0)
                            ix = -ix - 1;
                        else if (ix >= w)
                            ix = 2 * w - ix - 1;
                        sum += in[iy * w + ix] * obj.kdata[(ky + radius) * ksize + (kx + radius)];
                    }
                }
                out[y * w + x] = sum;
            }
        }
    }
    /**
     * @brief  锐化滤波器(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     */
    void sharpen(const float *in, float *out, const int w, const int h)
    {
        const filter obj = filter::sharpen();
        int kSize = obj.size;
        int radius = obj.radius;
#pragma omp parallel for
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    // 使用镜像边界
                    int iy = y + ky;
                    if (iy < 0)
                        iy = -iy - 1;
                    else if (iy >= h)
                        iy = 2 * h - iy - 1;
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        // 使用镜像边界
                        int ix = x + kx;
                        if (ix < 0)
                            ix = -ix - 1;
                        else if (ix >= w)
                            ix = 2 * w - ix - 1;
                        sum += in[iy * w + ix] * obj.kdata[(ky + radius) * kSize + (kx + radius)];
                    }
                }
                out[y * w + x] = sum;
            }
        }
    }
    /**
     * @brief sobel卷积(cpu omp)
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param dx x方向卷积
     * @param dy y方向卷积
     */
    void sobel(const float *in, float *out, const int w, const int h)
    {
        filter kernelX = filter::sobelX();
        filter kernelY = filter::sobelY();
        int kSize = kernelX.size;
        int radius = kernelX.size / 2;
#pragma omp parallel for
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                float gx = 0, gy = 0;
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    // 使用镜像边界
                    int iy = y + ky;
                    if (iy < 0)
                        iy = -iy - 1;
                    else if (iy >= h)
                        iy = 2 * h - iy - 1;
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        int ix = x + kx;
                        if (ix < 0)
                            ix = -ix - 1;
                        else if (ix >= w)
                            ix = 2 * w - ix - 1;

                        float pixel = in[iy * w + ix];
                        int kIndex = (ky + radius) * kSize + (kx + radius);

                        gx += pixel * kernelX.kdata[kIndex];
                        gy += pixel * kernelY.kdata[kIndex];
                    }
                }
                out[y * w + x] = ::sqrt(gx * gx + gy * gy);
            }
        }
    }
}