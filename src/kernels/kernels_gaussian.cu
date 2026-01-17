#include "kernels/kernels.cuh"
#include <cstdint>
#include "cuda/cuda_memory.hpp"
#include "filters/filter.hpp"
__global__ void gaussianConvolutionGlobal2D(const float *__restrict__ input, float *__restrict__ output,
                                            const int width, const int height, const float *kernel , const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = kSize >> 1;
    if (x >= width || y >= height)
        return;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky)
    {
        // 使用镜像边界
        int iy = y + ky;
        // 镜像处理
        if (iy < 0)
            iy = -iy - 1; // 镜像：0 → -1 → 0, -1 → -2 → 1
        else if (iy >= height)
            iy = 2 * height - iy - 1; // 镜像：h → h-1, h+1 → h-2

        for (int kx = -radius; kx <= radius; ++kx)
        {
            // 使用镜像边界
            int ix = x + kx;
            // 镜像处理
            if (ix < 0)
                ix = -ix - 1;
            else if (ix >= width)
                ix = 2 * width - ix - 1;
            // int ix = min(max(x + kx, 0), width - 1);
            // int iy = min(max(y + ky, 0), height - 1);
            sum += input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }
    output[y * width + x] = sum;
}

__global__ void gaussianConvolutionShared2D(const float *__restrict__ input, float *__restrict__ output,
                                            const int width, const int height, const int kSize)
{
    int radius = kSize >> 1;
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 计算共享内存大小（考虑边缘填充）    
    int tileWidth = blockDim.x + 2 * radius;
    int tileHeight = blockDim.y + 2 * radius;
    int tileSize = tileWidth * tileHeight;// 边缘填充
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

__global__ void gaussianHorizontalGlobal1D(const float *__restrict__ input, float *__restrict__ d_output,
                                           int width, int height, int kSize)
{
    const int radius = kSize >> 1;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.f;
#pragma unroll
    for (int k = -radius; k <= radius; ++k)
    {
        int gx = min(max(x + k, 0), width - 1); // clamp
        sum += input[y * width + gx] * constkernel[k + radius];
    }

    d_output[y * width + x] = sum;
}

__global__ void gaussianVerticalGlobal1D(const float *__restrict__ input, float *__restrict__ output,
                                         int width, int height, int kSize)
{
    const int radius = kSize >> 1;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.f;
#pragma unroll
    for (int k = -radius; k <= radius; ++k)
    {
        int gy = min(max(y + k, 0), height - 1); // clamp
        sum += input[gy * width + x] * constkernel[k + radius];
    }

    output[y * width + x] = sum;
}

__global__ void gaussianHorizontalShared1D(const float *__restrict__ input, float *__restrict__ output,
                                           int width, int height, int kSize)
{
    int radius = kSize >> 1;
    extern __shared__ float tile[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    int tx = threadIdx.x;
    int gx = x - radius;
    gx = max(0, min(gx, width - 1));
    if (tx < blockDim.x + 2 * radius)
    {
        int ix = blockIdx.x * blockDim.x - radius + tx;
        ix = max(0, min(ix, width - 1));
        tile[tx] = input[y * width + ix];
    }

    __syncthreads();

    // 计算卷积
    if (x < width && y < height)
    {
        float sum = 0.0f;

#pragma unroll
        for (int k = -radius; k <= radius; ++k)
        {
            sum += tile[tx + k + radius] * constkernel[k + radius];
        }

        output[y * width + x] = sum;
    }
}
__global__ void gaussianVerticalShared1D(const float *__restrict__ input, float *__restrict__ output,
                                         int width, int height, int kSize)
{
    int radius = kSize >> 1;

    extern __shared__ float tile[];

    int x = blockIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int ty = threadIdx.y;
    int gx = x - radius;
    gx = max(0, min(gx, width - 1));

    if (ty < blockDim.y + 2 * radius)
    {
        int iy = blockIdx.y * blockDim.y - radius + ty;
        iy = max(0, min(iy, height - 1));
        tile[ty] = input[iy * width + x];
    }
    __syncthreads();
    if (x < width && y < height)
    {
        float sum = 0.f;
#pragma unroll
        for (int k = -radius; k <= radius; ++k)
        {
            sum += tile[ty + k + radius] * constkernel[k + radius];
        }
        output[y * width + x] = sum;
    }
}

void launchGaussianBlur(filter_pipeline &pipe, const float *in, float *out, mem_type type, const int ksize, const float sigma, int block_w, int block_h)
{
    // filter gaussianObj = filter::gaussian2D(ksize, sigma);
    filter gaussianObj = filter::gaussian2D(ksize, sigma);
    const int width = pipe.width;
    const int height = pipe.height;
    // cuda_event start, stop;
    cuda_memory<float> d_temp(width * height);
    pipe.d_input.copy_from_host_async(in, width * height, pipe.stream.get());
    dim3 block(block_w, block_h);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    if (type == mem_type::GLOBAL)
    {
        cuda_memory<float> d_kernel(ksize * ksize);
        d_kernel.copy_from_host_async(gaussianObj.kdata.data(), ksize * ksize, pipe.stream.get());
        // start.record();
        gaussianConvolutionGlobal2D<<<grid, block, 0, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height,d_kernel.data(), ksize);
        // stop.record();
        // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(constkernel, gaussianObj.kdata.data(), ksize * ksize * sizeof(float), 0, cudaMemcpyHostToDevice, pipe.stream.get()));
        int shraedSize = (block_w + 2 * gaussianObj.radius) * (block_h + 2 * gaussianObj.radius) * sizeof(float);
        // start.record();
        gaussianConvolutionShared2D<<<grid, block, shraedSize, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, ksize);
        // stop.record();
        // cudaEventSynchronize(stop);// 等待事件完成
    }
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    pipe.d_output.copy_to_host_async(out, width * height, pipe.stream.get());
    return;
}
__device__ __forceinline__ uint8_t to_uchar(float v){
    v = fminf(fmaxf(v,0.0f),1.0f);
    return static_cast<uint8_t>(v * 255.0f);
}
__global__ void gaussian_rgba_kernel(const float* __restrict__ in, uchar4 * __restrict__ out, int w, int h){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y + w + x;
    float v = in[idx];
    uint8_t c = to_uchar(v);
    out[idx] = make_uchar4(c,c,c,255);
}



#include "pipeline/gl_frame_slot.hpp"
void gaussianRGBAGPU(gl_frame_slot &pipe, const float * in, int width, int height)
{
    pipe.pbo->map(pipe.stream.get());
    float* d_in;
    cudaMalloc(&d_in, width * height * sizeof(float));
    cudaMemcpyAsync(&d_in, in, sizeof(float) * width * height, cudaMemcpyHostToDevice, pipe.stream.get());
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16,(height + 15) / 16);

    // gaussian_rgba_kernel<<<grid, block, 0, slot->get_stream()>>>(static_cast<uint8_t *>(dev_ptr), width, height );
    // gaussian_rgba_kernel<<<grid, block, 0, pipe.stream.get()>>>(d_in, pipe.pbo->device_ptr(),width,height);
    cudaMemsetAsync(pipe.pbo->device_ptr(),0,pipe.pbo->size_bytes(),pipe.stream.get());

}