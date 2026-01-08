/**
 * @brief 是对convolution_kernel.cuh的文件函数实现
 */
//=========================================共享内存+常量内存===============================================
#include <cstdio>
// 卷积核放入常量内存（最快）
__constant__ float constkernel[4096];
__constant__ float c_sobel_dx[3] = {-1, 0, 1};
__constant__ float c_sobel_sm[3] = { 1, 2, 1};
#include <cuda_runtime.h>
/**
 * @brief 自定义卷积
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param ksize 内核大小
 */
__global__ void conv2dGlobalKernelWithShared(const float* __restrict__ input, float* __restrict__ output,
     const int width, const int height, const int kSize)
{
    int radius = kSize / 2;
    extern  __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //tile 尺寸 = block + halo
    int tileWidth   = blockDim.x + 2 * radius; 
    int tileHeight  = blockDim.y + 2 * radius; 
    int tileSize = tileWidth * tileHeight;  
    //边缘填充
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileSize; idx += blockDim.x * blockDim.y) {
        int iy = (blockIdx.y * blockDim.y - radius + idx / tileWidth);
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
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算共享内存大小（考虑边缘填充）
    int tileWidth   = blockDim.x + 2 * radius; 
    int tileHeight  = blockDim.y + 2 * radius; 
    int tileSize = tileWidth * tileHeight;  
    //边缘填充
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileSize; idx += blockDim.x * blockDim.y) {
        int iy = (blockIdx.y * blockDim.y - radius + idx / tileWidth);
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
 * @brief sobel算子 X方向
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 * @return __global__ 
 */
__global__ void sobelXConvolutionWithShared(const float* __restrict__ input,float* __restrict__ output,int width, int height)
{
    extern __shared__ float tile[];
    int TILE_PITCH = blockDim.x + 2;

    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y;

    if (y < height) {
        tile[ty * TILE_PITCH + tx] =
            (x < width) ? input[y * width + x] : 0.f;

        if (threadIdx.x == 0)
            tile[ty * TILE_PITCH] =
                (x > 0) ? input[y * width + x - 1] : 0.f;

        if (threadIdx.x == blockDim.x - 1)
            tile[ty * TILE_PITCH + tx + 1] =
                (x + 1 < width) ? input[y * width + x + 1] : 0.f;
    }

    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.f;
        #pragma unroll
        for (int k = -1; k <= 1; k++)
            sum += tile[ty * TILE_PITCH + tx + k] * c_sobel_dx[k + 1];

        output[y * width + x] = sum;
    }
}
/**
 * @brief sobel算子 Y方向
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 * @return __global__ 
 */
__global__ void sobelYConvolutionWithShared(const float* __restrict__ input,float* __restrict__ output,int width, int height)
{
    extern __shared__ float tile[];
    int TILE_PITCH = blockDim.x;

    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y + 1;

    if (x < width) {
        tile[ty * TILE_PITCH + tx] =(y < height) ? input[y * width + x] : 0.f;

        if (threadIdx.y == 0)
            tile[tx] =(y > 0) ? input[(y - 1) * width + x] : 0.f;

        if (threadIdx.y == blockDim.y - 1)
            tile[(ty + 1) * TILE_PITCH + tx] =(y + 1 < height) ? input[(y + 1) * width + x] : 0.f;
    }

    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.f;
        #pragma unroll
        for (int k = -1; k <= 1; k++)
            sum += tile[(ty + k) * TILE_PITCH + tx] * c_sobel_sm[k + 1];

        output[y * width + x] = sum;
    }
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
    int radius = kSize / 2;
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算共享内存大小（考虑边缘填充）
    int tileWidth   = blockDim.x + 2 * radius; 
    int tileHeight  = blockDim.y + 2 * radius; 
    int tileSize = tileWidth * tileHeight;  
    //边缘填充
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileSize; idx += blockDim.x * blockDim.y) {
        int iy = (blockIdx.y * blockDim.y - radius + idx / tileWidth);
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
    int radius = kSize / 2;
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算共享内存大小（考虑边缘填充）
    int tileWidth   = blockDim.x + 2 * radius; 
    int tileHeight  = blockDim.y + 2 * radius; 
    int tileSize = tileWidth * tileHeight;  
    //边缘填充
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileSize; idx += blockDim.x * blockDim.y) {
        int iy = (blockIdx.y * blockDim.y - radius + idx / tileWidth);
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
    int radius = kSize / 2;
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算共享内存大小（考虑边缘填充）
    int tileWidth   = blockDim.x + 2 * radius; 
    int tileHeight  = blockDim.y + 2 * radius; 
    int tileSize = tileWidth * tileHeight;  
    //边缘填充
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileSize; idx += blockDim.x * blockDim.y) {
        int iy = (blockIdx.y * blockDim.y - radius + idx / tileWidth);
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