#pragma once
#include <cuda_runtime.h>
extern __constant__ float constkernel[4096];
cudaError_t memCpyConstant(const float* hostKernel,int kernelSize){
    return cudaMemcpyToSymbol(constkernel, hostKernel, kernelSize, 0, cudaMemcpyHostToDevice);
}

