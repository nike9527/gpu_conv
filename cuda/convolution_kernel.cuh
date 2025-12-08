__constant__ float gauss_kernel[4096];  // 64×64 = 4096
__global__ void conv2d_global_kernel(const float* input, float* output, const int w, const int h, const float * kernel, const int ksize);

__global__ void gaussianConvolution(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * const kernel, const int kSize);

__global__ void sobelConvolution(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * const kernelX, const float * const kernelY, const int kSize);