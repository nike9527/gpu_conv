#include "core/gpu_timer.hpp"
#include "cuda/cuda_memory.hpp"
#include "bench_types.hpp"
#include "bench_config.hpp"
#include "bench_filter.hpp"
#include "core/triple_pipeline.hpp"
#include <vector>
#include <numeric>
#include <chrono>

extern cudaError_t memCpyConstant(const float *hostKernel, int kernelSize);
constexpr int PIPE_N = 3;
triple_pipeline<float, PIPE_N> pipeline(width *height);
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

/**
 * @brief Triple pipeline测试
 *  Triple vs Triple 差距 ≥ 1.3× 才是正常的（PCIe + kernel overlap）
 * @param hIn
 * @param hOut
 * @param w
 * @param h
 * @param kSize
 * @param filter
 * @param mtype
 * @param iters
 */
void launchConvTripleBuffer(const float *hIn, float *hOut, int w, int h, int kSize, filter_type filter, mem_type mtype, int iters)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int frame = 0; frame < num_frames; ++frame)
    {
        // === Producer ===
        auto &buf = pipeline.acquire();

        // 填 host input
        fill_input(buf.h_in(), width, height, frame);

        // 异步 H2D
        cudaMemcpyAsync(
            buf.d_in(), buf.h_in(),
            bytes,
            cudaMemcpyHostToDevice,
            buf.stream());

        // kernel
        launch_conv_kernel(
            buf.d_in(), buf.d_out(),
            width, height, ksize,
            buf.stream());

        // 异步 D2H
        cudaMemcpyAsync(
            buf.h_out(), buf.d_out(),
            bytes,
            cudaMemcpyDeviceToHost,
            buf.stream());

        pipeline.submit(buf);

        // === Consumer（非阻塞）===
        while (auto *done = pipeline.try_fetch())
        {
            consume_output(done->h_out(), width, height);
            pipeline.release(*done);
        }
    }
    // drain
    while (pipeline.inflight() > 0)
    {
        if (auto *done = pipeline.try_fetch())
        {
            consume_output(done->h_out(), width, height);
            pipeline.release(*done);
        }
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
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
        std::vector<float> hIn(c.width * c.height);
        std::vector<float> hOut(c.width * c.height);
        launchConvTripleBuffer(hIn.data(), hOut.data(), c.width, c.height, c.kSize, c.filter, c.mType, ITERS);
    }
    float totalMs = timer.toc(stream);
    float avgKernelMs = totalMs / ITERS;
    float gpixel = (float)(c.width * c.height) / (avgKernelMs * 1e-3f) / 1e9f;
    return {avgKernelMs, gpixel};
}