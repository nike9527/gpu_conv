__constant__ float gauss_kernel[4096];  // 64×64 = 4096
__global__ void conv2d_global_kernel(const float* input, float* output, int w, int h, const float* kernel, int ksize);

__global__ void gaussianConvolution(const float* __restrict__ input, float* __restrict__ output, int width, int height, const float* kernel, int kSize);