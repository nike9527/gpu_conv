
/**
 * @brief 是对convolution_kernel.cuh的文件函数实现
 */
//=========================================全局内存===============================================
#include <cuda_runtime.h>
/**
 * @brief 自定义卷积
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param kSize 内核大小
 */
__global__ void conv2dGlobalKernel(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float* const kernel, const int kSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int radius = kSize / 2;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            sum +=  input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }
    output[y * width + x] = sum;
}
/**
 * @brief 高斯模糊
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void gaussianConvolution(const float* __restrict__ input, float* __restrict__ output, 
                const int width, const int height, const float * const kernel, const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = kSize / 2;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        //使用镜像边界
        int iy = y + ky;
         // 镜像处理
        if (iy < 0) iy = -iy - 1; // 镜像：0 → -1 → 0, -1 → -2 → 1
        else if (iy >= height) iy = 2 * height - iy - 1; // 镜像：h → h-1, h+1 → h-2

        for (int kx = -radius; kx <= radius; ++kx) {
            // 使用镜像边界
            int ix = x + kx;
            // 镜像处理
            if (ix < 0)  ix = -ix - 1;
            else if (ix >= width) ix = 2*width - ix - 1;
            // int ix = min(max(x + kx, 0), width - 1);
            // int iy = min(max(y + ky, 0), height - 1);
            sum += input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }    
    output[y * width + x] = sum;
}
/**
 * @brief sobel算子
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 * @param kernelX 内核 X方向
 * @param kernelY 内核 Y方向
 * @param kSize 内核大小
 * @return __global__ 
 */
__global__ void sobelConvolution(const float* __restrict__ input, float* __restrict__ output, 
    const int width, const int height, const float * const kernelX, const float * const kernelY, const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= width || x >= height) return;

    float gx = 0, gy = 0;
    for (int ky = -1; ky <= 1; ++ky){
        //使用镜像边界
        int iy = y + ky;
        if (iy < 0) iy = -iy - 1;
        else if (iy >= height) iy = 2 * height - iy - 1;
        for (int kx = -1; kx <= 1; ++kx){
            int ix = x + kx;
            if (ix < 0)  ix = -ix - 1;
            else if (ix >= width) ix = 2*width - ix - 1;

            float pixel = input[iy * width + ix];
            int kIndex = (ky + 1) * kSize + (kx + 1);

            gx += (kernelX ? pixel * kernelX[kIndex] : 0);
            gy += (kernelY ? pixel * kernelY[kIndex] : 0);
        }
    }
    output[y * width + x] = ::sqrt(gx * gx + gy * gy);
}
/**
 * @brief 锐化滤波器
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void sharpenConvolution(const float* __restrict__ input, float* __restrict__ output, 
    const int width, const int height, const float * kernel, const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = kSize / 2;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        // 使用镜像边界
        int iy = y + ky;
        if (iy < 0) iy = -iy - 1; 
        else if (iy >= height) iy = 2 * height - iy - 1;
        for (int kx = -radius; kx <= radius; ++kx) {
            // 使用镜像边界
            int ix = x + kx;
            if (ix < 0)  ix = -ix - 1;
            else if (ix >= width) ix = 2*width - ix - 1;
            sum += input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }
    output[y * width + x] = sum;
}
/**
 * @brief 均值模糊
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void meanBlurConvolution(const float* __restrict__ input, float* __restrict__ output, 
    const int width, const int height, const float * kernel, const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = kSize / 2;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        // 使用镜像边界
        int iy = y + ky;
        if (iy < 0) iy = -iy - 1; 
        else if (iy >= height) iy = 2 * height - iy - 1;
        for (int kx = -radius; kx <= radius; ++kx) {
            // 使用镜像边界
            int ix = x + kx;
            if (ix < 0)  ix = -ix - 1;
            else if (ix >= width) ix = 2*width - ix - 1;
            sum += input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }
    output[y * width + x] = sum;
}
/**
 * @brief 拉普拉斯算子
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void laplacianConvolution(const float* __restrict__ input, float* __restrict__ output, 
    const int width, const int height, const float * kernel, const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = kSize / 2;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        // 使用镜像边界
        int iy = y + ky;
        if (iy < 0) iy = -iy - 1; 
        else if (iy >= height) iy = 2 * height - iy - 1;
        for (int kx = -radius; kx <= radius; ++kx) {
            // 使用镜像边界
            int ix = x + kx;
            if (ix < 0)  ix = -ix - 1;
            else if (ix >= width) ix = 2*width - ix - 1;
            sum += input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }
    output[y * width + x] = sum;
}