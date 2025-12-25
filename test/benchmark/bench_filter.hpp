#include "convolution_kernel.cuh"
#include "kernel.hpp"
#include <cuda_runtime.h>
#include <iostream>

void benchGaussianConvolutionGPU(const float* d_in, float* d_out, const int w, const int h, const int kSize, const float sigma,int block_w = 16, int block_h = 16) {

}