#include "core/gpu_timer.hpp"
#include "cuda/cuda_memory.hpp"
#include "bench_types.hpp"
#include "bench_config.hpp"
#include "bench_filter.hpp"
#include "convolution_kernel.cuh"
#include "core/triple_pipeline.hpp"
#include <vector>
#include <numeric>

extern cudaError_t memCpyConstant(const float *hostKernel, int kernelSize);
// ====== 你需要对接的接口（自己实现） ======
void launchConvSingleStream(const float *d_in, float *d_out, int width, int height, int kSize, filter_type filter_type, mem_type mtype, float *d_kernel, int block_w = 16, int block_h = 16)
{
    dim3 block(block_w, block_h);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    switch (filter_type)
    {
    case filter_type::SOBEL:
    {
        filter sobelX = filter::sobelX();
        filter sobelY = filter::sobelY();
        float *d_kernelX = nullptr;
        float *d_kernelY = nullptr;
        cudaMalloc(&d_kernelX, kSize * kSize * sizeof(float));
        cudaMalloc(&d_kernelY, kSize * kSize * sizeof(float));
        cudaMemcpy(d_kernelX, sobelX.kdata.data(), kSize * kSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernelY, sobelY.kdata.data(), kSize * kSize * sizeof(float), cudaMemcpyHostToDevice);

        if (mtype == mem_type::GLOBAL)
        {
            sobelConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernelX, d_kernelY, kSize);
        }
        else if (mtype == mem_type::SHAREDCONST)
        {
            int shraedSize = block_w + (kSize / 2) * block_h + (kSize / 2);
            sobelConvolution<<<grid, block, shraedSize>>>(d_in, d_out, width, height, d_kernelX, d_kernelY, kSize);
        }
        cudaFree(d_kernelX);
        cudaFree(d_kernelY);
    }
    break;
    case filter_type::GAUSSIAN:
        if (mtype == mem_type::GLOBAL)
        {
            gaussianConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernel, kSize);
        }
        else if (mtype == mem_type::SHAREDCONST)
        {
            int shraedSize = (block_w + (2 * kSize / 2)) * (block_h + (2 * kSize / 2)) * sizeof(float);
            gaussianConvolutionWithShared<<<grid, block, shraedSize>>>(d_in, d_out, width, height, kSize);
        }
        break;
    case filter_type::MEANBLUR:
        if (mtype == mem_type::GLOBAL)
        {
            meanBlurConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernel, kSize);
        }
        else if (mtype == mem_type::SHAREDCONST)
        {
            int shraedSize = (block_w + (2 * kSize / 2)) * (block_h + (2 * kSize / 2)) * sizeof(float);
            meanBlurConvolutionWithShared<<<grid, block, shraedSize>>>(d_in, d_out, width, height, kSize);
        }
        break;
    case filter_type::SHARPEN:
        if (mtype == mem_type::GLOBAL)
        {
            sharpenConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernel, kSize);
        }
        else if (mtype == mem_type::SHAREDCONST)
        {
            int shraedSize = (block_w + (2 * kSize / 2)) * (block_h + (2 * kSize / 2)) * sizeof(float);
            sharpenConvolutionWithShared<<<grid, block, shraedSize>>>(d_in, d_out, width, height, kSize);
        }

        break;
    case filter_type::LAPLACIAN:
        if (mtype == mem_type::GLOBAL)
        {
            laplacianConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernel, kSize);
        }
        else if (mtype == mem_type::SHAREDCONST)
        {
            int shraedSize = (block_w + (2 * kSize / 2)) * (block_h + (2 * kSize / 2)) * sizeof(float);
            laplacianConvolutionWithShared<<<grid, block, shraedSize>>>(d_in, d_out, width, height, kSize);
        }
        break;
    default:
        break;
    }
}

void launchConvTripleBuffer(const float *hIn, float *hOut, int w, int h, int kSize, filter_type filter, mem_type mtype, int iters)
{
    triple_pipeline<float> pipe(w * h);
    auto &buf = pipe.acquire();
    int bytes = w * h * sizeof(float);
    std::memcpy(buf.h_in(), hIn, bytes);
    cudaMemcpyAsync(buf.d_in(), buf.h_in(), bytes, cudaMemcpyHostToDevice, buf.stream());
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    int shraedSize = (block.x + (2 * kSize / 2)) * (block.y + (2 * kSize / 2)) * sizeof(float);
    sharpenConvolutionWithShared<<<grid, block, shraedSize, buf.stream()>>>(buf.d_in(), buf.d_out(), w, h, kSize);
    cudaMemcpyAsync(buf.h_out(), buf.d_out(), bytes, cudaMemcpyDeviceToHost, buf.stream());

    pipe.submit(buf);
    auto *done = pipe.try_fetch();
    // done->h_out();
}
// ===========================================

BenchResult runBenchmark(const BenchCase &c)
{
    const int WARMUP = 10;
    const int ITERS = 100;
    gpu_timer timer;
    size_t size = c.width * c.height;
    cuda_memory<float> dIn(size), dOut(size);
    cuda_memory<float> d_kernel(c.kSize * c.kSize);
    filter kernel = getFilter(c.filter, c.kSize);
    if (c.mType == mem_type::GLOBAL)
    {
        size_t dSize = c.kSize * c.kSize;
        d_kernel.copy_from_host(kernel.kdata.data(), dSize);
    }
    else if (c.mType == mem_type::SHAREDCONST)
    {
        CUDA_CHECK(memCpyConstant(kernel.kdata.data(), c.kSize * c.kSize * sizeof(float)));
    }
    // warm-up
    for (int i = 0; i < WARMUP; ++i)
    {
        launchConvSingleStream(dIn.data(), dOut.data(), c.width, c.height, c.kSize, c.filter, c.mType, d_kernel.data());
    }
    cudaDeviceSynchronize();
    cuda_stream stream;
    timer.tic(stream);
    if (c.pipeline == PipelineType::SINGLE_STREAM)
    {
        for (int i = 0; i < ITERS; ++i)
        {
            launchConvSingleStream(dIn.data(), dOut.data(), c.width, c.height, c.kSize, c.filter, c.mType, d_kernel.data());
        }
    }
    else
    {
        // triple-buffer 通常按帧算，这里简化成 ITERS 帧
        // std::vector<float> hIn(c.width * c.height);
        // std::vector<float> hOut(c.width * c.height);
        // launchConvTripleBuffer(hIn.data(), hOut.data(),c.width, c.height, c.kSize,c.filter, c.mType, ITERS);

        // ===== 正确的 triple-pipeline benchmark =====
        /**
         * @brief
         * 1 Nsight Systems
         *      多 stream overlap 是否存在
         *   2 人为加大 H2D / D2H
         *      triple pipeline 吞吐是否高于 single
         *   3 把 N=2 / 3 / 4
         *       stall 点是否按预期变化
         *
         */
        const size_t elems = c.width * c.height;
        const size_t bytes = elems * sizeof(float);
        std::vector<float> hIn(elems);
        std::vector<float> hOut(elems);
        triple_pipeline<float> pipe(elems);
        int submitted = 0;
        int completed = 0;
        dim3 block(16, 16);
        dim3 grid((c.width + block.x - 1) / block.x,
                  (c.height + block.y - 1) / block.y);
        int sharedSize =
            (block.x + (2 * c.kSize / 2)) *
            (block.y + (2 * c.kSize / 2)) *
            sizeof(float);
        while (completed < ITERS)
        {
            // ---- submit stage ----
            if (submitted < ITERS)
            {
                /**
                 * @brief  stage 可以自然扩展成多段
                 *
                 * std::vector<pipeline_stage<float>*> stages = {&stage1,&stage2,&stage3};

                    for (auto* s : stages)
                        s->enqueue(buf);
                 *
                 */
                sharpen_stage sharpen(c.width, c.height, c.kSize);
                auto &buf = pipe.acquire();
                std::memcpy(buf.h_in(), hIn.data(), bytes);
                cudaMemcpyAsync(buf.d_in(), buf.h_in(), bytes,
                                cudaMemcpyHostToDevice, buf.stream());
                // sharpenConvolutionWithShared<<<grid, block, sharedSize, buf.stream()>>>( buf.d_in(), buf.d_out(), c.width, c.height, c.kSize);
                sharpen.enqueue(buf);
                cudaMemcpyAsync(buf.h_out(), buf.d_out(), bytes,
                                cudaMemcpyDeviceToHost, buf.stream());
                pipe.submit(buf);
                ++submitted;
            }
            // ---- fetch stage ----
            if (auto *done = pipe.try_fetch())
            {
                std::memcpy(hOut.data(), done->h_out(), bytes);
                ++completed;
            }
        }
    }
    float totalMs = timer.toc(stream);
    float avgKernelMs = totalMs / ITERS;
    float gpixel = (float)(c.width * c.height) / (avgKernelMs * 1e-3f) / 1e9f;
    return {avgKernelMs, gpixel};
}