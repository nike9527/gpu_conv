#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误在 %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_KERNEL_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "核函数错误: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
    


constexpr int NUM_BUFFERS = 3;
struct bufferStream {
    float* h_in;
    float* h_out;
    float* d_in;
    float* d_out;
    cudaStream_t stream;
    cudaEvent_t e;   //完成事件
};
struct hostImage {
    int width;
    int height;
    float* data;   // host pointer
};
struct deviceImage {
    int width;
    int height;
    float* d_data; // device pointer
};
void initializeDeviceMemory(bufferStream (&devMem)[NUM_BUFFERS],int size){
    for(int i=0; i<NUM_BUFFERS; i++){
        cudaMallocHost(&devMem[i].h_in, size);
        cudaMallocHost(&devMem[i].h_out, size);
        cudaMalloc(&devMem[i].d_in, size);
        cudaMalloc(&devMem[i].d_out, size);
        cudaStreamCreate(&devMem[i].stream);
        cudaEventCreateWithFlags(&devMem[i].e, cudaEventDisableTiming);
    }
}

void freeDeviceMemory(bufferStream (&devMem)[NUM_BUFFERS]){
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        cudaStreamDestroy(devMem[i].stream);
        cudaEventDestroy(devMem[i].e);
        cudaFree(devMem[i].d_in);
        cudaFree(devMem[i].d_out);
        cudaFreeHost(devMem[i].h_in);
        cudaFreeHost(devMem[i].h_out);
    }
}
