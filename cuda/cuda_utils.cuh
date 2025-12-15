#pragma once
#include <cuda_runtime.h>
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
