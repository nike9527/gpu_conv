#pragma once
#include <cuda_runtime.h>
#include "cuda_stream.hpp"
#include "cuda_event.hpp"
constexpr int NUM_BUFFERS = 3;
struct bufferStream {
    float* h_in;
    float* h_out;
    float* d_in;
    float* d_out;
    cuda_stream stream;
    cuda_event e;   //完成事件
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
