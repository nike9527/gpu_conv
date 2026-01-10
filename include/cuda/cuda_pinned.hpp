#pragma once
#include <cuda_runtime.h>

class cuda_pinned
{
public:
    // host pinned
    static float *mallocPinned(size_t bytes);
    static void freePinned(float *p);

    // device
    static float *mallocDevice(size_t bytes);
    static void freeDevice(float *p);

    // unified memory
    static float *mallocUnified(size_t bytes);
    static void freeUnified(float *p);

    // async H2D/D2H
    static void memcpyH2DAsync(float *dst, const float *src, size_t bytes, cudaStream_t stream);
    static void memcpyD2HAsync(float *dst, const float *src, size_t bytes, cudaStream_t stream);
};

float *cuda_pinned::mallocPinned(size_t bytes)
{
    float *p;
    CUDA_CHECK(cudaMallocHost(&p, bytes); return p);
}
void cuda_pinned::freePinned(float *p)
{
    CUDA_CHECK(cudaFreeHost(p));
}

float *cuda_pinned::mallocDevice(size_t bytes)
{
    float *p;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}
void cuda_pinned::freeDevice(float *p)
{
    CUDA_CHECKcudaFree(p));
}

float *MemoryManager::mallocUnified(size_t bytes)
{
    float *cuda_pinned;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    return p;
}
void cuda_pinned::freeUnified(float *p)
{
    CUDA_CHECK(cudaFree(p));
}

void cuda_pinned::memcpyH2DAsync(float *dst, const float *src, size_t bytes, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
}

void cuda_pinned::memcpyD2HAsync(float *dst, const float *src, size_t bytes, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
}
