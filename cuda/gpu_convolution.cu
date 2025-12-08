#pragma once
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "kernel.hpp"
#include "convolution_kernel.cuh"

void gaussianConvolutionGPU(const float* in, float* out, const int w, const int h, const int kSize, const float sigma) {
    Kernel gaKernel = Kernel::gaussian(kSize,sigma);
    size_t n = size_t(w) * size_t(h) * sizeof(float);
    float *d_in=nullptr, *d_out=nullptr, *d_kernel=nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_in, n); if (err != cudaSuccess) { 
        std::cerr<<"cudaMalloc d_in failed\n"; 
        return; 
    }
    err = cudaMalloc(&d_out, n); if (err != cudaSuccess) { 
        cudaFree(d_in); std::cerr<<"cudaMalloc d_out failed\n"; 
        return; 
    }
    err = cudaMalloc(&d_kernel, kSize*kSize*sizeof(float)); 
    if (err != cudaSuccess) { 
        cudaFree(d_in); 
        cudaFree(d_out); 
        std::cerr<<"cudaMalloc d_kernel failed\n"; 
        return; 
    }

    err = cudaMemcpy(d_in, in, n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_in failed\n"; 
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernel);
    }
    err = cudaMemcpy(d_kernel, gaKernel.kdata.data(), kSize*kSize*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_kernel failed\n"; 
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernel);
    }

    dim3 block(16,16);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    auto t1 = std::chrono::high_resolution_clock::now();
    gaussianConvolution<<<grid, block>>>(d_in, d_out, w, h, d_kernel, kSize);
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
    return;
}

void sobelConvolutionGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy){
   
    size_t n = size_t(w) * size_t(h) * sizeof(float);
    float *d_in=nullptr, *d_out=nullptr, *d_kernelX=nullptr, *d_kernelY=nullptr;
    Kernel sobelX = Kernel::sobelX();
    Kernel sobelY = Kernel::sobelY();
    int kSize = sobelX.size;
    cudaError_t err;
    err = cudaMalloc(&d_in, n); if (err != cudaSuccess) { 
        std::cerr<<"cudaMalloc d_in failed\n"; 
        return; 
    }
    err = cudaMemcpy(d_in, in, n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_in failed\n"; 
        cudaFree(d_in); 
        return;
    }
    err = cudaMalloc(&d_out, n); if (err != cudaSuccess) { 
        cudaFree(d_in); std::cerr<<"cudaMalloc d_out failed\n"; 
        return; 
    }
    if(dx != 0){
        err = cudaMalloc(&d_kernelX, kSize*kSize*sizeof(int)); 
        if (err != cudaSuccess) { 
            cudaFree(d_in); 
            cudaFree(d_out); 
            std::cerr<<"cudaMalloc d_kernel failed\n"; 
            return; 
        }
        err = cudaMemcpy(d_kernelX, sobelX.kdata.data(), kSize*kSize*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { 
            std::cerr<<"cudaMemcpy d_kernel failed\n"; 
            cudaFree(d_in); 
            cudaFree(d_out); 
            cudaFree(d_kernelX);
            return;
        }
    }
    if(dy != 0){
        err = cudaMalloc(&d_kernelY, kSize*kSize*sizeof(int)); 
        if (err != cudaSuccess) { 
            cudaFree(d_in); 
            cudaFree(d_out); 
            cudaFree(d_kernelX);
            std::cerr<<"cudaMalloc d_kernel failed\n"; 
            return; 
        }

        err = cudaMemcpy(d_kernelY, sobelY.kdata.data(), kSize*kSize*sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { 
            std::cerr<<"cudaMemcpy d_kernel failed\n"; 
            cudaFree(d_in); 
            cudaFree(d_out); 
            cudaFree(d_kernelX);
            cudaFree(d_kernelY);
            return;
        }
    }
    dim3 block(16,16);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    auto t1 = std::chrono::high_resolution_clock::now();
    sobelConvolution<<<grid, block>>>(d_in, d_out, w, h, d_kernelX, d_kernelY, kSize);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    err = cudaGetLastError(); 
    if (err != cudaSuccess) { 
        std::cerr<<"kernel launch failed: "<<cudaGetErrorString(err)<<"\n"; 
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernelX);
        cudaFree(d_kernelY);
    }

    err = cudaMemcpy(out, d_out, n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_out failed\n";
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernelX);
        cudaFree(d_kernelY);
    }

    cudaFree(d_in); 
    cudaFree(d_out); 
    cudaFree(d_kernelX);
    cudaFree(d_kernelY);
    return;
} 

void sharpenConvolutionGPU(const float* in, float* out, const int w, const int h){
    Kernel sharpenKernel = Kernel::sharpen();
    int kSize = sharpenKernel.size;
    size_t n = size_t(w) * size_t(h) * sizeof(float);
    float *d_in=nullptr, *d_out=nullptr, *d_kernel=nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_in, n); if (err != cudaSuccess) { 
        std::cerr<<"cudaMalloc d_in failed\n"; 
        return; 
    }
    err = cudaMalloc(&d_out, n); if (err != cudaSuccess) { 
        cudaFree(d_in); std::cerr<<"cudaMalloc d_out failed\n"; 
        return; 
    }
    err = cudaMalloc(&d_kernel, kSize*kSize*sizeof(float)); 
    if (err != cudaSuccess) { 
        cudaFree(d_in); 
        cudaFree(d_out); 
        std::cerr<<"cudaMalloc d_kernel failed\n"; 
        return; 
    }

    err = cudaMemcpy(d_in, in, n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_in failed\n"; 
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernel);
    }
    err = cudaMemcpy(d_kernel, sharpenKernel.kdata.data(), kSize*kSize*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        std::cerr<<"cudaMemcpy d_kernel failed\n"; 
        cudaFree(d_in); 
        cudaFree(d_out); 
        cudaFree(d_kernel);
    }

    dim3 block(16,16);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    auto t1 = std::chrono::high_resolution_clock::now();
    sharpenConvolution<<<grid, block>>>(d_in, d_out, w, h, d_kernel, kSize);
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
    return;
}