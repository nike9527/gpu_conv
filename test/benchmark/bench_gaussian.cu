#include "bench_config.hpp"
#include "core/gpu_timer.hpp"
#include "core/filter_pipeline.hpp"
#include "filters/kernel_desc.hpp"
#include "kernels/kernels.cuh"
#include <benchmark/benchmark.h>
static void BM_Gaussian_SharedConst(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    int ksize = state.range(2);
    int block_w = state.range(3);
    int block_h = state.range(4);
    float sigma = state.range(5);
    dim3 block(block_w, block_h);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    filter gaussianObj = filter::gaussian2D(ksize, sigma);
    filter_pipeline pipe(width, height);
    cudaMemcpyToSymbolAsync(constkernel, gaussianObj.kdata.data(), ksize * ksize * sizeof(float), 0, cudaMemcpyHostToDevice, pipe.stream.get());
    int shraedSize = (block_w + 2 * gaussianObj.radius) * (block_h + 2 * gaussianObj.radius) * sizeof(float);
    // 预热（非常重要）
    for (int i = 0; i < 5; ++i)
    {
        gaussianConvolutionShared2D<<<grid, block, shraedSize, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, ksize);
    }
    cudaDeviceSynchronize();

    gpu_timer timer;
    double kernel_ms = 0.0;

    for (auto _ : state)
    {
        timer.tic(pipe.stream);
        gaussianConvolutionShared2D<<<grid, block, shraedSize, pipe.stream.get()>>>(pipe.d_input.data(), pipe.d_output.data(), width, height, ksize);
        kernel_ms += timer.toc(pipe.stream);
    }

    kernel_ms /= state.iterations();

    double pixels = double(width) * height;
    double gpixel_per_s = pixels / (kernel_ms * 1e6);

    state.counters["Kernel_ms"] = kernel_ms;
    state.counters["GPixel/s"] = benchmark::Counter(gpixel_per_s, benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Gaussian_SharedConst)->Args({1920, 1080, 3, 16, 16, 5})->Args({3840, 2160, 3, 16, 16, 5})->UseRealTime()->Unit(benchmark::kMillisecond);
