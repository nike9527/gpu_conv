#include <cuda_runtime.h>
#include <math.h>
__global__ void sobelSeparableX(const float *__restrict__ input, float *__restrict__ temp,
                                int width, int height, const float *__restrict__ kRow, const float *__restrict__ kCol, int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float gx = 0.0f;

    // 先在列方向做卷积 (垂直)
    for (int ky = -kSize / 2; ky <= kSize / 2; ++ky)
    {
        int iy = y + ky;
        if (iy < 0)
            iy = -iy - 1;
        else if (iy >= height)
            iy = 2 * height - iy - 1;

        gx += input[iy * width + x] * kCol[ky + kSize / 2];
    }

    temp[y * width + x] = gx;
}

__global__ void sobelSeparableY(const float *__restrict__ temp, float *__restrict__ output,
                                int width, int height, const float *__restrict__ kRow, const float *__restrict__ kCol, int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float gx = 0.0f;

    // 再在行方向做卷积 (水平)
    for (int kx = -kSize / 2; kx <= kSize / 2; ++kx)
    {
        int ix = x + kx;
        if (ix < 0)
            ix = -ix - 1;
        else if (ix >= width)
            ix = 2 * width - ix - 1;

        gx += temp[y * width + ix] * kRow[kx + kSize / 2];
    }

    output[y * width + x] = gx;
}

__global__ void sobelSeparable(
    const float *__restrict__ input,
    float *__restrict__ output,
    int width,
    int height,
    const float *__restrict__ kRowX, const float *__restrict__ kColX,
    const float *__restrict__ kRowY, const float *__restrict__ kColY,
    int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float gx = 0.0f;
    float gy = 0.0f;

    // --- 分离卷积：先垂直卷积（列方向） ---
    float tempX = 0.0f;
    float tempY = 0.0f;

    for (int ky = -kSize / 2; ky <= kSize / 2; ++ky)
    {
        int iy = y + ky;
        if (iy < 0)
            iy = -iy - 1; // 镜像边界
        else if (iy >= height)
            iy = 2 * height - iy - 1;

        float pixel = input[iy * width + x];

        tempX += pixel * kColX[ky + kSize / 2];
        tempY += pixel * kColY[ky + kSize / 2];
    }

    // --- 再水平卷积（行方向） ---
    for (int kx = -kSize / 2; kx <= kSize / 2; ++kx)
    {
        int ix = x + kx;
        if (ix < 0)
            ix = -ix - 1;
        else if (ix >= width)
            ix = 2 * width - ix - 1;

        gx += tempX * kRowX[kx + kSize / 2];
        gy += tempY * kRowY[kx + kSize / 2];
    }

    output[y * width + x] = sqrtf(gx * gx + gy * gy);
}

#define KSIZE 3 // Sobel kernel size

// Sobel 核一维分量，放入常量内存
__constant__ float h_kRowX[KSIZE] = {1, 0, -1};
__constant__ float h_kColX[KSIZE] = {1, 2, 1};
__constant__ float h_kRowY[KSIZE] = {1, 2, 1};
__constant__ float h_kColY[KSIZE] = {1, 0, -1};
template <int BLOCK_W, int BLOCK_H>
__global__ void sobelShared(
    const float *__restrict__ input,
    float *__restrict__ output,
    int width, int height)
{
    // block 内线程坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 全局像素坐标
    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;

    // 分配共享内存，扩大边界以支持卷积
    __shared__ float shMem[BLOCK_H + KSIZE - 1][BLOCK_W + KSIZE - 1];

    // 共享内存坐标
    int shX = tx + KSIZE / 2;
    int shY = ty + KSIZE / 2;

    // --- 加载共享内存（带镜像边界） ---
    for (int dy = -KSIZE / 2; dy <= KSIZE / 2; dy += BLOCK_H)
    {
        for (int dx = -KSIZE / 2; dx <= KSIZE / 2; dx += BLOCK_W)
        {
            int ix = x + dx;
            int iy = y + dy;

            // 镜像边界
            if (ix < 0)
                ix = -ix - 1;
            else if (ix >= width)
                ix = 2 * width - ix - 1;
            if (iy < 0)
                iy = -iy - 1;
            else if (iy >= height)
                iy = 2 * height - iy - 1;

            int sX = shX + dx;
            int sY = shY + dy;

            if (sX >= 0 && sX < BLOCK_W + KSIZE - 1 &&
                sY >= 0 && sY < BLOCK_H + KSIZE - 1)
            {
                shMem[sY][sX] = input[iy * width + ix];
            }
        }
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    // --- 分离卷积 ---
    float tempX = 0.0f, tempY = 0.0f;

    // 垂直卷积
    for (int ky = -KSIZE / 2; ky <= KSIZE / 2; ++ky)
    {
        tempX += shMem[shY + ky][shX] * d_kColX[ky + KSIZE / 2];
        tempY += shMem[shY + ky][shX] * d_kColY[ky + KSIZE / 2];
    }

    float gx = 0.0f, gy = 0.0f;
    // 水平卷积
    for (int kx = -KSIZE / 2; kx <= KSIZE / 2; ++kx)
    {
        gx += tempX * d_kRowX[kx + KSIZE / 2];
        gy += tempY * d_kRowY[kx + KSIZE / 2];
    }

    output[y * width + x] = sqrtf(gx * gx + gy * gy);
}

void launchSobel(filter_pipeline &pipe, const float *in, float *out, mem_type type, int block_w, int block_h)
{
    // float *d_temp;
    // cudaMalloc(&d_temp, width * height * sizeof(float));

    // // 对 Gx
    // sobelSeparableX<<<grid, block>>>(d_input, d_temp, width, height, kRowX, kColX, 3);
    // sobelSeparableY<<<grid, block>>>(d_temp, d_outputGx, width, height, kRowX, kColX, 3);

    // // 对 Gy
    // sobelSeparableX<<<grid, block>>>(d_input, d_temp, width, height, kRowY, kColY, 3);
    // sobelSeparableY<<<grid, block>>>(d_temp, d_outputGy, width, height, kRowY, kColY, 3);

    // // 最终梯度幅值
    // computeMagnitude<<<grid, block>>>(d_outputGx, d_outputGy, d_output, width, height);
    //===========================================================================================
    // dim3 block(16, 16);
    // dim3 grid((width + 15) / 16, (height + 15) / 16);

    // sobelShared<16, 16><<<grid, block>>>(d_input, d_output, width, height);
    // cudaDeviceSynchronize();
    //===========================================================================================

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sobelSeparable<<<grid, block>>>(
        d_input, d_output,
        width, height,
        d_kRowX, d_kColX,
        d_kRowY, d_kColY,
        3);
    cudaDeviceSynchronize();
    return;
}