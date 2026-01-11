#include "kernels/kernels.cuh"
#include "cuda/cuda_memory.hpp"
#include "filters/filter.hpp"

__global__ void conv2dKernelGlobal(const float *__restrict__ input, float *__restrict__ output,
                                   const int width, const int height, const float *const kernel, const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int radius = kSize >> 1;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ky++)
    {
        for (int kx = -radius; kx <= radius; kx++)
        {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            sum += input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }
    output[y * width + x] = sum;
}

__global__ void conv2dKernelShared(const float *__restrict__ input, float *__restrict__ output,
                                   const int width, const int height, const int kSize)
{
    int radius = kSize >> 1;
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // tile 尺寸 = block + halo
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

void launchFilter(filter_pipeline &pipe, const float *in, float *out, mem_type type, const filter &filterObj, int block_w, int block_h)
{
    // cuda_event start, stop;
    const int width = pipe.width;
    const int height = pipe.height;
    const int ksize = filterObj.size;
    const int radius = filterObj.radius;
    dim3 block(block_w, block_h);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    pipe.d_input.copy_from_host_async(in, width * height, pipe.stream.get());
    if (type == mem_type::GLOBAL)
    {
        static thread_local cuda_memory<float> d_kernel(ksize * ksize);
        d_kernel.copy_from_host_async(filterObj.kdata.data(), ksize * ksize, pipe.stream.get());
        // start.record()
        conv2dKernelGlobal<<<grid, block, 0, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, d_kernel.data(), ksize);
        // stop.record();
        // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(constkernel, filterObj.kdata.data(), ksize * ksize * sizeof(float), 0, cudaMemcpyHostToDevice, pipe.stream.get()));
        int shraedSize = (block_w + 2 * radius) * (block_h + 2 * radius) * sizeof(float);
        // start.record()
        conv2dKernelShared<<<grid, block, shraedSize, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, ksize);
        // stop.record();
        // cudaEventSynchronize(stop);// 等待事件完成
    }
    CHECK_KERNEL_ERROR();
    //std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    pipe.d_output.copy_to_host_async(out, width * height, pipe.stream.get());
    return;
}