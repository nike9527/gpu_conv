#include "bench_timer.hpp"
#include "bench_types.hpp"
#include "bench_config.hpp"
#include "bench_filter.hpp"
#include "convolution_gpu.hpp"
#include "convolution_kernel.cuh"
#include "kernel.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <numeric>


// ====== 你需要对接的接口（自己实现） ======
void launchConvSingleStream(const float* d_in, float* d_out,int width, int height, int kSize,FilterType filter, MemType mtype, float* d_kernel, int block_w=16,int block_h=16){
    dim3 block(block_w,block_h);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    switch (filter) {
        case FilterType::SOBEL:{
                Kernel sobelX = Kernel::sobelX();
                Kernel sobelY = Kernel::sobelY();
                float* d_kernelX=nullptr;
                float* d_kernelY=nullptr;
                cudaMalloc(&d_kernelX, kSize*kSize*sizeof(float));
                cudaMalloc(&d_kernelY, kSize*kSize*sizeof(float));
                cudaMemcpy(d_kernelX, sobelX.kdata.data(), kSize*kSize*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_kernelY, sobelY.kdata.data(), kSize*kSize*sizeof(float), cudaMemcpyHostToDevice);

                if(mtype == MemType::GLOBAL){
                    sobelConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernelX, d_kernelY, kSize);
                }else if(mtype == MemType::SHARED_CONST){
                    int shraedSize = block_w + (kSize/2) * block_h + (kSize/2);
                    sobelConvolution<<<grid, block, shraedSize>>>(d_in, d_out, width, height, d_kernelX, d_kernelY, kSize);
                }
                cudaFree(d_kernelX);
                cudaFree(d_kernelY);
                break;
            }
        case FilterType::GAUSSIAN:
            if(mtype == MemType::GLOBAL){
                gaussianConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernel, kSize);
            }else if(mtype == MemType::SHARED_CONST){
                int shraedSize = (block_w + (2 * kSize/2)) * (block_h + (2 * kSize/2)) * sizeof(float);
                gaussianConvolutionWithShared<<<grid, block, shraedSize>>>(d_in, d_out, width, height, kSize);
            }
            break;
        case FilterType::MEAN:
            if(mtype == MemType::GLOBAL){
                meanBlurConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernel, kSize);
            }else if(mtype == MemType::SHARED_CONST){
                int shraedSize = (block_w + (2 * kSize/2)) * (block_h + (2 * kSize/2)) * sizeof(float);
                meanBlurConvolutionWithShared<<<grid, block,shraedSize>>>(d_in, d_out, width, height, kSize);
            }
            break;
        case FilterType::SHARPEN:
            if(mtype == MemType::GLOBAL){
                sharpenConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernel, kSize);
            }else if(mtype == MemType::SHARED_CONST){
                int shraedSize = (block_w + (2 * kSize/2)) * (block_h + (2 * kSize/2)) * sizeof(float);
                sharpenConvolutionWithShared<<<grid, block, shraedSize>>>(d_in, d_out, width, height, kSize);
            }

            break;
        case FilterType::LAPLACIAN:
            if(mtype == MemType::GLOBAL){
                laplacianConvolution<<<grid, block>>>(d_in, d_out, width, height, d_kernel, kSize);
            }else if(mtype == MemType::SHARED_CONST){
                int shraedSize = (block_w + (2 * kSize/2)) * (block_h + (2 * kSize/2)) * sizeof(float);
                laplacianConvolutionWithShared<<<grid, block,shraedSize>>>(d_in, d_out, width, height, kSize);
            }
            break;
        default:
            break;
    }
}

void launchConvTripleBuffer(const float* hIn, float* hOut,int w, int h, int kSize,FilterType filter, MemType mtype,int iters){

}
// ===========================================

BenchResult runBenchmark(const BenchCase& c) {
    const int WARMUP = 10;
    const int ITERS  = 100;
    cudaError_t err = cudaError::cudaSuccess;
    size_t bytes = c.width * c.height * sizeof(float);
    float* dIn;  
    float* d_kernel = nullptr;
    err = cudaMalloc(&dIn, bytes);
    if (err != cudaSuccess) {
        printf("---cudaMalloc 失败: %s\n", cudaGetErrorString(err));
    }
    float* dOut; 
    err = cudaMalloc(&dOut, bytes);
    if (err != cudaSuccess) {
        printf("===cudaMalloc 失败: %s\n", cudaGetErrorString(err));
    }

    Kernel kernel = getKernel(c.filter);
    if( c.mType == MemType::GLOBAL){
        err = cudaMalloc(&d_kernel, c.kSize*c.kSize*sizeof(float));
        if (err != cudaSuccess) {
            printf("===cudaMalloc kernel: %s\n", cudaGetErrorString(err));
        }
        err = cudaMemcpy(d_kernel, kernel.kdata.data(), c.kSize*c.kSize*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("===cudaMemcpy d_kernel: %s\n", cudaGetErrorString(err));
        }
    }else if( c.mType == MemType::SHARED_CONST){
        uploadKernelToConstant(kernel.kdata.data(),c.kSize*c.kSize*sizeof(float));
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // warm-up
    for (int i = 0; i < WARMUP; ++i) {
        launchConvSingleStream(dIn, dOut, c.width, c.height,c.kSize, c.filter, c.mType, d_kernel);
    }
    cudaDeviceSynchronize();
    GpuTimer timer;
    timer.tic(stream);
    // if (c.pipeline == PipelineType::SINGLE_STREAM) {
        for (int i = 0; i < ITERS; ++i) {
            launchConvSingleStream(dIn, dOut, c.width, c.height,c.kSize, c.filter, c.mType, d_kernel);
        }
    // } else {
    //     // triple-buffer 通常按帧算，这里简化成 ITERS 帧
    //     std::vector<float> hIn(c.width * c.height);
    //     std::vector<float> hOut(c.width * c.height);
    //     launchConvTripleBuffer(hIn.data(), hOut.data(),c.width, c.height, c.kSize,c.filter, c.mType, ITERS);
    // }
    float totalMs = timer.toc(stream);
    float avgKernelMs = totalMs / ITERS;
    float gpixel =(float)(c.width * c.height) /(avgKernelMs * 1e-3f) / 1e9f;
    cudaFree(dIn);
    cudaFree(dOut);
    cudaFree(d_kernel);
    cudaStreamDestroy(stream);

    return { avgKernelMs, gpixel };
}