#include "kernels/kernels.cuh"
#include "cuda/cuda_memory.hpp"
#define KSIZE 3
// Sobel 核一维分量，放入常量内存
__constant__ float d_kRowX[3] = {1.0f, 0.f, -1.0f};
__constant__ float d_kColX[3] = {1.0f, 2.0f, 1.0f};
__constant__ float d_kRowY[3] = {1.0f, 2.0f, 1.0f};
__constant__ float d_kColY[3] = {1.0f, 0.f, -1.0f};


__global__ void sobelConvolutionGlobal(const float *__restrict__ input, float *__restrict__ output, int width, int height, int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // --- 方法A：直接2D卷积（简单但正确）---
    float gx = 0.0f, gy = 0.0f;

    for (int ky = -1; ky <= 1; ky++)
    {
        int iy = y + ky;
        // 边界处理
        if (iy < 0)
            iy = 0;
        else if (iy >= height)
            iy = height - 1;

        for (int kx = -1; kx <= 1; kx++)
        {
            int ix = x + kx;
            if (ix < 0)
                ix = 0;
            else if (ix >= width)
                ix = width - 1;

            float p = input[iy * width + ix];

            // 注意：这里实际上是 kCol[ky] * kRow[kx]
            gx += p * d_kColX[ky + 1] * d_kRowX[kx + 1];
            gy += p * d_kColY[ky + 1] * d_kRowY[kx + 1];
        }
    }

    output[y * width + x] = sqrtf(gx * gx + gy * gy);
}

__global__ void sobelConvolutionShared(const float *__restrict__ input, float *__restrict__ output, int width, int height, int ksize)
{
    int radius = ksize / 2;
    extern __shared__ float smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // 全局像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算共享内存大小（考虑边缘填充）    
    int tileWidth = blockDim.x + 2 * radius;
    int tileHeight = blockDim.y + 2 * radius;
    int tileSize = tileWidth * tileHeight;

    float *tile = smem;
    float *vertX = tile + tileWidth * tileHeight;
    float *vertY = vertX + blockDim.x * blockDim.y;
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileSize; idx += blockDim.x * blockDim.y)
    {
        int iy = (blockIdx.y * blockDim.y - radius + idx / tileWidth);
        int ix = (blockIdx.x * blockDim.x - radius + idx % tileWidth);
        // clamp
        ix = min(max(ix, 0), width - 1);
        iy = min(max(iy, 0), height - 1);
        tile[idx] = input[iy * width + ix];
    }
    __syncthreads();
    if (x >= width || y >= height)
        return;

    // --- 垂直卷积 ---
    int cx = tx + radius;
    int cy = ty + radius;
    float vX = 0.0f, vY = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky)
    {
        float p = tile[(cy + ky) * tileWidth + cx];
        vX += p * d_kColX[ky + radius];
        vY += p * d_kColY[ky + radius];
    }
    int vIdx = ty * blockDim.x + tx;
    vertX[vIdx] = vX;
    vertY[vIdx] = vY;
    __syncthreads();

    float gx = 0.0f, gy = 0.0f;
    // --- 水平卷积 ---
    for (int kx = -radius; kx <= radius; ++kx)
    {
        int nx = min(max(tx + kx, 0), blockDim.x - 1);
        int nIdx = ty * blockDim.x + nx;

        gx += vertX[nIdx] * d_kRowX[kx + radius];
        gy += vertY[nIdx] * d_kRowY[kx + radius];
    }
    output[y * width + x] = sqrtf(gx * gx + gy * gy);
}

void launchSobel(filter_pipeline &pipe, const float *in, float *out, mem_type type, int block_w, int block_h)

{
    const int width = pipe.width;
    const int height = pipe.height;
    // cuda_event start, stop;
    pipe.d_input.copy_from_host_async(in, width * height, pipe.stream.get());
    dim3 block(block_w, block_h);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    if (type == mem_type::GLOBAL)
    {
        // start.record();
        sobelConvolutionGlobal<<<grid, block, 0, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, KSIZE);
        // stop.record();
        // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
        int tileBytes = (block.x + 2) * (block.y + 2) * sizeof(float); // tile
        int vertBytes = 2 * block.x * block.y * sizeof(float);         // vertX + vertY
        int shraedSize = tileBytes + vertBytes;
        // start.record();
        sobelConvolutionShared<<<grid, block, shraedSize, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, KSIZE);
        // stop.record();
        // cudaEventSynchronize(stop);// 等待事件完成
    }
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    pipe.d_output.copy_to_host_async(out, width * height, pipe.stream.get());
    return;
}