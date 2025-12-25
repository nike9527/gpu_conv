#pragma once
#include <cuda_runtime.h>
#include <cstdio>
struct GpuTimer {
    cudaEvent_t start, stop;
    cudaError_t err = cudaError::cudaSuccess;
    GpuTimer() {
       err =  cudaEventCreate(&start);
        if (err != cudaSuccess) {
            printf("cudaEventCreate start: %s\n", cudaGetErrorString(err));
        }
        err =  cudaEventCreate(&stop);
        if (err != cudaSuccess) {
            printf("cudaEventCreate stop: %s\n", cudaGetErrorString(err));
        }
    }
    ~GpuTimer() {
        err =  cudaEventDestroy(start);
        if (err != cudaSuccess) {
            printf("cudaEventDestroy start: %s\n", cudaGetErrorString(err));
        }
        err = cudaEventDestroy(stop);
        if (err != cudaSuccess) {
            printf("cudaEventDestroy stop: %s\n", cudaGetErrorString(err));
        }
    }

    void tic(cudaStream_t stream = 0) {
        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            printf("cudaEventRecord : %s\n", cudaGetErrorString(err));
        }
    }

    float toc(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
        cudaError_t err = cudaError::cudaSuccess;
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            printf("cudaEventSynchronize stop: %s\n", cudaGetErrorString(err));
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};
