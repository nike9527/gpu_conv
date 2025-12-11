#include <cstdio>
//=========================================共享内存+常量内存===============================================
// 卷积核放入常量内存（最快）
extern __constant__ float constkernel[4096];

#include <cuda_runtime.h>
/**
 * @brief 自定义卷积
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param ksize 内核大小
 */
__global__ void conv2d_global_kernelWithShared(const float* input, float* output, const int width, const int height, const int ksize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >=  height) return;
    int r = ksize / 2;
    float sum = 0.0f;
    for (int ky = -r; ky <= r; ky++) {
        for (int kx = -r; kx <= r; kx++) {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            sum += input[iy * width + ix] * constkernel[(ky + r) * ksize + (kx + r)];
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
 * @param ksize 内核大小
 */

 __global__ void gaussianConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, 
                const int width, const int height, const int kSize)
{
    int radius = kSize / 2;
    __shared__ float tile[18 * 18];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算共享内存大小（考虑边缘填充）
    int tileWidth   = blockDim.x + 2 * radius; 
    int tileHeight  = blockDim.y + 2 * radius; 
    
    int tileSize = tileWidth * tileHeight;  
    //边缘填充
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileSize; idx += blockDim.x * blockDim.y) {
        int iy = (blockIdx.y * blockDim.y - radius + idx / tileHeight);
        int ix = (blockIdx.x * blockDim.x - radius + idx % tileWidth);
        // clamp
        ix = min(max(ix, 0), width - 1);
        iy = min(max(iy, 0), height - 1);
        tile[idx] = input[iy * width + ix];
    }        
    __syncthreads();
    
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                sum += tile[(threadIdx.y + radius + ky) * tileWidth + (threadIdx.x + radius + kx)] * constkernel[(ky + radius) * kSize + (kx + radius)];
            }
        }
        output[y * width + x] = sum;
    }
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
__global__ void sobelConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, 
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
 * @param ksize 内核大小
 */
__global__ void sharpenConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, 
    const int width, const int height, const int kSize)
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
            sum += input[iy * width + ix] * constkernel[(ky + radius) * kSize + (kx + radius)];
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
 * @param ksize 内核大小
 */
__global__ void meanBlurConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, 
    const int width, const int height, const int kSize)
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
            sum += input[iy * width + ix] * constkernel[(ky + radius) * kSize + (kx + radius)];
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
 * @param ksize 内核大小
 */
__global__ void laplacianConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, 
    const int width, const int height, const int kSize)
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
            sum += input[iy * width + ix] * constkernel[(ky + radius) * kSize + (kx + radius)];
        }
    }
    output[y * width + x] = sum;
}