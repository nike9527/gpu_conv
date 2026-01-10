#include "kernels/kernels.cuh"
#include "cuda/cuda_memory.hpp"
#include "filters/filter.hpp"


__global__ void meanBlurConvolutionGlobal(const float *__restrict__ input, float *__restrict__ output,
                                          const int width, const int height, const float *kernel, const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = kSize / 2;
    if (x >= width || y >= height)
        return;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky)
    {
        // 使用镜像边界
        int iy = y + ky;
        if (iy < 0)
            iy = -iy - 1;
        else if (iy >= height)
            iy = 2 * height - iy - 1;
        for (int kx = -radius; kx <= radius; ++kx)
        {
            // 使用镜像边界
            int ix = x + kx;
            if (ix < 0)
                ix = -ix - 1;
            else if (ix >= width)
                ix = 2 * width - ix - 1;
            sum += input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }
    output[y * width + x] = sum;
}


__global__ void meanBlurConvolutionShared(const float *__restrict__ input, float *__restrict__ output,
                                          const int width, const int height, const int kSize)
{
    int radius = kSize / 2;
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算共享内存大小（考虑边缘填充）
    int tileWidth = blockDim.x + 2 * radius;
    int tileHeight = blockDim.y + 2 * radius;
    int tileSize = tileWidth * tileHeight;
    // 边缘填充
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
    if (x < width && y < height)
    {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky)
        {
            for (int kx = -radius; kx <= radius; ++kx)
            {
                sum += tile[(threadIdx.y + radius + ky) * tileWidth + (threadIdx.x + radius + kx)] * constkernel[(ky + radius) * kSize + (kx + radius)];
            }
        }
        output[y * width + x] = sum;
    }
}


void launchMeanBlur(filter_pipeline &pipe, const float *in, float *out, mem_type type, int ksize, int block_w, int block_h)

{
    filter meanBlurObj = filter::meanBlur(ksize);
    const int width = pipe.width;
    const int height = pipe.height;
    const int radius = meanBlurObj.radius;
    // cuda_event start, stop;
    pipe.d_input.copy_from_host_async(in, width * height, pipe.stream.get());
    dim3 block(block_w, block_h);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    if (type == mem_type::GLOBAL)
    {
        cuda_memory<float> d_kernel(ksize * ksize);
        d_kernel.copy_from_host_async(meanBlurObj.kdata.data(), ksize * ksize, pipe.stream.get());
        // start.record();
        meanBlurConvolutionGlobal<<<grid, block, 0, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, d_kernel.data(), ksize);
        // stop.record();
        // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(constkernel, meanBlurObj.kdata.data(), ksize * ksize * sizeof(float), 0, cudaMemcpyHostToDevice, pipe.stream.get()));
        int shraedSize = (block_w + 2 * radius) * (block_h + 2 * radius) * sizeof(float);
        // start.record();
        meanBlurConvolutionShared<<<grid, block, shraedSize, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, ksize);
        // stop.record();
        // cudaEventSynchronize(stop);// 等待事件完成
    }

    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    pipe.d_output.copy_to_host_async(out, width * ksize, pipe.stream.get());
    return;
}