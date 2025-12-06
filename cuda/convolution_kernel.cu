#include <cuda_runtime.h>
__global__ void conv2d_global_kernel(const float* input, float* output, int w, int h, const float* kernel, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int r = ksize / 2;
    float sum = 0.0f;
    for (int ky = -r; ky <= r; ky++) {
        for (int kx = -r; kx <= r; kx++) {
            int ix = min(max(x + kx, 0), w - 1);
            int iy = min(max(y + ky, 0), h - 1);
            sum += input[iy * w + ix] * kernel[(ky + r) * ksize + (kx + r)];
        }
    }
    output[y * w + x] = sum;
}

__global__ void gaussianConvolution(const float* __restrict__ input, float* __restrict__ output,
                                  int width, int height, const float* kernel, int kSize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = kSize / 2;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        //使用镜像边界
        int iy = y + ky;
         // 镜像处理
        if (iy < 0) iy = -iy - 1; // 镜像：0 → -1 → 0, -1 → -2 → 1
        else if (iy >= height) iy = 2 * height - iy - 1; // 镜像：h → h-1, h+1 → h-2

        for (int kx = -radius; kx <= radius; ++kx) {
            // 使用镜像边界
            int ix = x + kx;
            // 镜像处理
            if (ix < 0)  ix = -ix - 1;
            else if (ix >= width) ix = 2*width - ix - 1;
            // int ix = min(max(x + kx, 0), width - 1);
            // int iy = min(max(y + ky, 0), height - 1);
            sum += input[iy * width + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
        }
    }    
    output[y * width + x] = sum;
}