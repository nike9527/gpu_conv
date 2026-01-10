#pragma once
#include <cuda_runtime.h>
#include "cuda/cuda_stream.hpp"
#include "filters/kernel_desc.hpp"
#include "core/filter_pipeline.hpp"
#include "filters/filter.hpp"

extern __constant__ float constkernel[4096];
cudaError_t memCpyConstant(const float* hostKernel,int kernelSize);
//=============================kernel配置函数=======================================
void launchFilter(filter_pipeline &pipe, const float *in, float *out, mem_type type, const filter &filterObj, int block_w = 16, int block_h = 16);
void launchGaussianBlur(filter_pipeline &pipe, const float *in, float *out, mem_type type, const int ksize = 3, const float sigma = 1.f, int block_w = 16, int block_h = 16);
void launchLaplacian(filter_pipeline &pipe, const float *in, float *out, mem_type type, int block_w = 16, int block_h = 16);
void launchMeanBlur(filter_pipeline &pipe, const float *in, float *out, mem_type type,  int ksize = 3, int block_w = 16, int block_h = 16);
void launchsharpen(filter_pipeline &pipe, const float *in, float *out, mem_type type, int block_w = 16, int block_h = 16);
void launchSobel(filter_pipeline &pipe, const float *in, float *out, mem_type type, int block_w = 16, int block_h = 16);



