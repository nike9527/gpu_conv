/**
 * CPU baseline + OpenMP
 */
#include "cpu_convolution.hpp"
#include "kernel.hpp"
#include <algorithm>
#include <omp.h>
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
void conv2d_cpu_omp(const float* in, float* out, const int w, const int h, const float * const kernel, int const ksize) {
    int r = ksize / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int ky = -r; ky <= r; ky++) {
                for (int kx = -r; kx <= r; kx++) {
                    int ix = std::min(std::max(x + kx, 0), w - 1);
                    int iy = std::min(std::max(y + ky, 0), h - 1);
                    sum += in[iy * w + ix] * kernel[(ky + r) * ksize + (kx + r)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}
/**
 * @brief 高斯卷积
 * @param in  输入数据
 * @param out   输出数据
 * @param w  宽度
 * @param h  高度
 * @param kSize  核大小
 */
void gaussianConvolution(const float* in, float* out, const int w, const int h, const int kSize, const float sigma){
    Kernel kernel = Kernel::gaussian(kSize,sigma);
    int radius = kSize / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
                // 使用镜像边界
                int iy = y + ky;
                 // 镜像处理
                // if (iy < 0) iy = -iy;
                // else if (iy >= h) iy = h - 1;
                if (iy < 0) iy = -iy - 1; // 镜像：0 → -1 → 0, -1 → -2 → 1
                else if (iy >= h) iy = 2 * h - iy - 1; // 镜像：h → h-1, h+1 → h-2
                for (int kx = -radius; kx <= radius; ++kx) {
                    // 使用镜像边界
                    int ix = x + kx;
                     // 镜像处理
                    // if (ix < 0) ix = -ix;
                    // else if (ix >= w) ix = w - 1;
                    if (ix < 0)  ix = -ix - 1;
                    else if (ix >= w) ix = 2*w - ix - 1;
                    sum += in[iy * w + ix] * kernel.kdata[(ky + radius) * kSize + (kx + radius)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}
/**
 * @brief sobel卷积
 * @param in  输入数据
 * @param out   输出数据
 * @param w  宽度
 * @param h  高度
 * @param dx x方向卷积
 * @param dy y方向卷积
 */
void sobelConvolution(const float* in, float* out, const int w, const int h,const int dx, const int dy)
{
    Kernel kernelX = Kernel::sobelX();
    Kernel kernelY = Kernel::sobelY();
    int kSize = kernelX.size;
    int radius = kernelX.size /  2;
    #pragma omp parallel for
    for (int y = 0; y < h; ++y){
        for (int x = 0; x < w; ++x){
            float gx = 0, gy = 0;
            for (int ky = -radius; ky <= radius; ++ky){
                //使用镜像边界
                int iy = y + ky;
                if (iy < 0) iy = -iy - 1;
                else if (iy >= h) iy = 2 * h - iy - 1;
                for (int kx = -radius; kx <= radius; ++kx){
                    int ix = x + kx;
                    if (ix < 0)  ix = -ix - 1;
                    else if (ix >= w) ix = 2*w - ix - 1;

                    float pixel = in[iy * w + ix];
                    int kIndex = (ky + radius) * kSize + (kx + radius);

                    gx += (dx ? pixel * kernelX.kdata[kIndex] : 0);
                    gy += (dy ? pixel * kernelY.kdata[kIndex] : 0);
                }
            }
            out[y * w + x] = ::sqrt(gx * gx + gy * gy);
        }
    }
}

/**
 * @brief  锐化滤波器
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 */
void sharpenConvolution(const float* in, float* out, const int w, const int h){
    Kernel kernel = Kernel::sharpen();
    int kSize = kernel.size;
    int radius = kSize / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
                // 使用镜像边界
                int iy = y + ky;
                if (iy < 0) iy = -iy - 1; 
                else if (iy >= h) iy = 2 * h - iy - 1;
                for (int kx = -radius; kx <= radius; ++kx) {
                    // 使用镜像边界
                    int ix = x + kx;
                    if (ix < 0)  ix = -ix - 1;
                    else if (ix >= w) ix = 2*w - ix - 1;
                    sum += in[iy * w + ix] * kernel.kdata[(ky + radius) * kSize + (kx + radius)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}

/**
 * @brief  均值模糊
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param kSize  核大小
 */
void meanBlurConvolution(const float* in, float* out, const int w, const int h,int const kSize){
    Kernel kernel = Kernel::meanBlur(kSize);
    int radius = kSize / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
                // 使用镜像边界
                int iy = y + ky;
                if (iy < 0) iy = -iy - 1; 
                else if (iy >= h) iy = 2 * h - iy - 1;
                for (int kx = -radius; kx <= radius; ++kx) {
                    // 使用镜像边界
                    int ix = x + kx;
                    if (ix < 0)  ix = -ix - 1;
                    else if (ix >= w) ix = 2*w - ix - 1;
                    sum += in[iy * w + ix] * kernel.kdata[(ky + radius) * kSize + (kx + radius)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}

void laplacianConvolution(const float* in, float* out, const int w, const int h){
    Kernel kernel = Kernel::laplacian();
    int kSize = kernel.size;
    int radius = kSize / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
                // 使用镜像边界
                int iy = y + ky;
                if (iy < 0) iy = -iy - 1; 
                else if (iy >= h) iy = 2 * h - iy - 1;
                for (int kx = -radius; kx <= radius; ++kx) {
                    // 使用镜像边界
                    int ix = x + kx;
                    if (ix < 0)  ix = -ix - 1;
                    else if (ix >= w) ix = 2*w - ix - 1;
                    sum += in[iy * w + ix] * kernel.kdata[(ky + radius) * kSize + (kx + radius)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}