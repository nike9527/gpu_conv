#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "kernel.hpp"
#include "convolution_kernel.cuh"
#include <vector>
#include "image.hpp"
#include "stream_buffer.hpp"
#include <chrono>
/**
 * @brief 外部函数调用和函数的入口(api)
 */
// 卷积核放入常量内存（最快）
__constant__ float constkernel[4096]; // 最大支持7x7卷积核
//=============================全局内存=======================================
void conv2dGlobalGPU(const float* in, float* out, const int w, const int h, const int kSize, const float* kernel,int block_w = 16, int block_h = 16){
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernel(kSize*kSize);
    d_input.copy_from_host(in, w * h);
    d_kernel.copy_from_host(kernel,kSize*kSize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    start.record();
    conv2dGlobalKernel<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
    stop.record();
    cudaEventSynchronize(stop);// 等待事件完成 
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
void gaussianConvolutionGPU(const float* in, float* out, const int w, const int h, const int kSize, const float sigma,int block_w = 16, int block_h = 16) {
    Kernel gaKernel = Kernel::gaussian(kSize,sigma);
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernel(kSize*kSize);
    d_input.copy_from_host(in, w * h);
    d_kernel.copy_from_host(gaKernel.kdata.data(),kSize*kSize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    start.record();
    gaussianConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
void sobelConvolutionGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy,int block_w = 16, int block_h = 16){
    Kernel sobelX = Kernel::sobelX();
    Kernel sobelY = Kernel::sobelY();
    int kSize = sobelX.size;
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernelX(kSize*kSize);
    cuda_memory<float> d_kernelY(kSize*kSize);
    d_input.copy_from_host(in, w * h);
    d_kernelX.copy_from_host(sobelX.kdata.data(),kSize*kSize);
    d_kernelY.copy_from_host(sobelY.kdata.data(),kSize*kSize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    start.record();
    sobelConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernelX.data(), d_kernelY.data(), kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
} 
void sharpenConvolutionGPU(const float* in, float* out, const int w, const int h,int block_w = 16, int block_h = 16){
    Kernel sharpenKernel = Kernel::sharpen();
    int kSize = sharpenKernel.size;
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernel(kSize*kSize);
    d_input.copy_from_host(in, w * h);
    d_kernel.copy_from_host(sharpenKernel.kdata.data(),kSize*kSize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    start.record();
    sharpenConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
void meanBlurConvolutionGPU(const float* in, float* out, const int w, const int h,int const kSize,int block_w = 16, int block_h = 16){
    Kernel meanKernel = Kernel::meanBlur(kSize);
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernel(kSize*kSize);
    d_input.copy_from_host(in, w * h);
    d_kernel.copy_from_host(meanKernel.kdata.data(),kSize*kSize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    start.record();
    meanBlurConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
void laplacianConvolutionGPU(const float* in, float* out, const int w, const int h,int block_w = 16, int block_h = 16){
    Kernel laplacianKernel = Kernel::laplacian();
    int kSize = laplacianKernel.size;
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernel(kSize*kSize);
    d_input.copy_from_host(in, w * h);
    d_kernel.copy_from_host(laplacianKernel.kdata.data(),kSize*kSize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    start.record();
    laplacianConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
//=========================================共享内存+常量内存===============================================
void conv2dWithSharedGPU(const float* in, float* out, const int w, const int h, const int kSize, const float* kernel,int block_w = 16, int block_h = 16){
    int r = kSize / 2;
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    cudaError_t err = cudaMemcpyToSymbol(constkernel, kernel, kSize*kSize*sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_kernel failed\n"; 
    }
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    int shraedSize = (block_w + 2*r) * (block_h + 2*r) * sizeof(float);
    start.record();
    conv2dGlobalKernelWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
void gaussianConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h, const int kSize, const float sigma,int block_w = 16, int block_h = 16) {
    Kernel gaKernel = Kernel::gaussian(kSize,sigma);
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    cudaError_t err = cudaMemcpyToSymbol(constkernel, gaKernel.kdata.data(), kSize*kSize*sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_kernel failed\n"; 
    }
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    int shraedSize = (block_w + 2*gaKernel.radius) * (block_h + 2*gaKernel.radius) * sizeof(float);
    start.record();
    gaussianConvolutionWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
void sobelConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy,int block_w = 16, int block_h = 16){
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    start.record();
    size_t shraedXSize = block.y * (block.x + 2) * sizeof(float);
    size_t shraedYSize = (block.y + 2) * block.x * sizeof(float);
    sobelXConvolutionWithShared<<<grid, block, shraedXSize>>>(d_input.data(), d_output.data(), w, h);
    sobelYConvolutionWithShared<<<grid, block, shraedYSize>>>(d_input.data(), d_output.data(), w, h);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
} 
void sharpenConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,int block_w = 16, int block_h = 16){
    Kernel sharpenKernel = Kernel::sharpen();
    int kSize = sharpenKernel.size;
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    cudaError_t err = cudaMemcpyToSymbol(constkernel, sharpenKernel.kdata.data(), kSize*kSize*sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_kernel failed\n"; 
    }
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    int shraedSize = (block_w + 2*sharpenKernel.radius) * (block_h + 2*sharpenKernel.radius) * sizeof(float);
    start.record();
    sharpenConvolutionWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
void meanBlurConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,int const kSize,int block_w = 16, int block_h = 16){
    Kernel meanKernel = Kernel::meanBlur(kSize);
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    cudaError_t err = cudaMemcpyToSymbol(constkernel, meanKernel.kdata.data(), kSize*kSize*sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_kernel failed\n";
    }
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    int shraedSize = (block_w + 2*meanKernel.radius) * (block_h + 2*meanKernel.radius) * sizeof(float);
    start.record();
    meanBlurConvolutionWithShared<<<grid, block,shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
void laplacianConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,int block_w = 16, int block_h = 16){
    Kernel laplacianKernel = Kernel::laplacian();
    int kSize = laplacianKernel.size;
    cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    cudaError_t err = cudaMemcpyToSymbol(constkernel, laplacianKernel.kdata.data(), kSize*kSize*sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_kernel failed\n";  
    }
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    int shraedSize = (block_w + 2*laplacianKernel.radius) * (block_h + 2*laplacianKernel.radius) * sizeof(float);
    start.record();
    laplacianConvolutionWithShared<<<grid, block,shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
    stop.record(); 
    cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}

//=========================================cuda stream===============================================
void conv2dWithAsyncGPU(std::vector<Image>& in,std::vector<Image>& out, const int kSize, const float* kernel,int block_w = 16, int block_h = 16){
    cudaMemcpyToSymbol(constkernel, kernel, kSize*kSize*sizeof(float), 0, cudaMemcpyHostToDevice);
    int size = 3840 * 2160 * sizeof(float);
    int r = kSize / 2;
    const int NUM_BUFFERS = 3;
    stream_buffer<float> buffs[NUM_BUFFERS] ={stream_buffer<float>(size), stream_buffer<float>(size),stream_buffer<float>(size)};
    for (int i = 0; i < in.size(); ++i){
        int cur = i % NUM_BUFFERS;
        stream_buffer<float>& buf = buffs[cur];
        if (i >= NUM_BUFFERS) {
            cudaEventSynchronize(buf.event());
        }
        cudaMemcpyAsync(buf.d_in(),in[i].data.data(),in[i].width * in[i].height * sizeof(float),cudaMemcpyHostToDevice,buf.stream());
        dim3 block(block_w,block_h);
        dim3 grid((in[i].width+block.x-1)/block.x, (in[i].height+block.y-1)/block.y);
        int shraedSize = (block_w + (2 * r)) * (block_h + (2 * r)) * sizeof(float);
        auto t1 = std::chrono::high_resolution_clock::now();
        conv2dGlobalKernelWithShared<<<grid, block, shraedSize, buf.stream()>>>(buf.d_in(), buf.d_out(), in[i].width, in[i].height, kSize);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        cudaMemcpyAsync(out[i].data.data(),buf.d_out(),out[i].width * out[i].height * sizeof(float),cudaMemcpyDeviceToHost, buf.stream());
        cudaEventRecord(buf.event(), buf.stream());
    }
    // 同步所有流
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        cudaEventSynchronize(buffs[i].event());
    }
}

void uploadKernelToConstant(const float* hostKernel,int kernelSize){
    cudaError_t err = cudaMemcpyToSymbol(constkernel,hostKernel,kernelSize, 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpyToSymbol failed: %s\n",cudaGetErrorString(err));
    }else{
        printf("cudaMemcpyToSymbol success: %s\n",cudaGetErrorString(err));
    }
}
