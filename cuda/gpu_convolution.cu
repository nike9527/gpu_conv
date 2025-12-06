#pragma once
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "convolution_kernel.cuh"

bool gaussianConvolutionGPU(const float* in, float* out, int w, int h, const float* kernel, int ksize) {
    size_t n = size_t(w) * size_t(h) * sizeof(float);
    float *d_in=nullptr, *d_out=nullptr, *d_kernel=nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_in, n); if (err != cudaSuccess) { 
        std::cerr<<"cudaMalloc d_in failed\n"; 
        return false; 
    }
    err = cudaMalloc(&d_out, n); if (err != cudaSuccess) { 
        cudaFree(d_in); std::cerr<<"cudaMalloc d_out failed\n"; 
        return false; 
    }
    err = cudaMalloc(&d_kernel, ksize*ksize*sizeof(float)); 
    if (err != cudaSuccess) { 
        cudaFree(d_in); 
        cudaFree(d_out); 
        std::cerr<<"cudaMalloc d_kernel failed\n"; 
        return false; 
    }

    err = cudaMemcpy(d_in, in, n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_in failed\n"; 
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernel);
    }
    err = cudaMemcpy(d_kernel, kernel, ksize*ksize*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_kernel failed\n"; 
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernel);
    }

    dim3 block(16,16);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    auto t1 = std::chrono::high_resolution_clock::now();
    gaussianConvolution<<<grid, block>>>(d_in, d_out, w, h, d_kernel, ksize);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    err = cudaGetLastError(); 
    if (err != cudaSuccess) { 
        std::cerr<<"kernel launch failed: "<<cudaGetErrorString(err)<<"\n"; 
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernel);
    }

    err = cudaMemcpy(out, d_out, n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_out failed\n";
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernel);
    }

    cudaFree(d_in); 
    cudaFree(d_out); 
    cudaFree(d_kernel);
    return true;
}
