#include "bench_timer.hpp"
#include "bench_types.hpp"
#include "bench_config.hpp"
#include "bench_filter.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
/**
 * @brief gaussianConvolution(const float* __restrict__ input, float* __restrict__ output, 
                const int width, const int height, const float * const kernel, const int kSize)
 */
// ====== 你需要对接的接口（自己实现） ======
void launchConvSingleStream(const float* dIn, float* dOut,int w, int h, int kSize,FilterType filter, MemType mtype){

}

void launchConvTripleBuffer(const float* hIn, float* hOut,int w, int h, int kSize,FilterType filter, MemType mtype,int iters){

}
// ===========================================

BenchResult runBenchmark(const BenchCase& c) {
    const int WARMUP = 10;
    const int ITERS  = 100;

    size_t bytes = c.width * c.height * sizeof(float);

    float* dIn;  cudaMalloc(&dIn, bytes);
    float* dOut; cudaMalloc(&dOut, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // warm-up
    for (int i = 0; i < WARMUP; ++i) {
        launchConvSingleStream(dIn, dOut, c.width, c.height,c.kSize, c.filter, c.mType);
    }
    cudaDeviceSynchronize();

    GpuTimer timer;
    timer.tic(stream);

    if (c.pipeline == PipelineType::SINGLE_STREAM) {
        for (int i = 0; i < ITERS; ++i) {
            launchConvSingleStream(dIn, dOut, c.width, c.height,c.kSize, c.filter, c.mType);
        }
    } else {
        // triple-buffer 通常按帧算，这里简化成 ITERS 帧
        std::vector<float> hIn(c.width * c.height);
        std::vector<float> hOut(c.width * c.height);
        launchConvTripleBuffer(hIn.data(), hOut.data(),c.width, c.height, c.kSize,c.filter, c.mType, ITERS);
    }

    float totalMs = timer.toc(stream);
    float avgKernelMs = totalMs / ITERS;

    float gpixel =
        (float)(c.width * c.height) /
        (avgKernelMs * 1e-3f) / 1e9f;

    cudaFree(dIn);
    cudaFree(dOut);
    cudaStreamDestroy(stream);

    return { avgKernelMs, gpixel };
}
