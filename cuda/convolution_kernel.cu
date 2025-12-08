#include <cuda_runtime.h>

__global__ void conv2d_global_kernel(const float* input, float* output, const int w, const int h, const float* const kernel, const int ksize){
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
                const int width, const int height, const float * const kernel, const int kSize)
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

__global__ void sobelConvolution(const float* __restrict__ input, float* __restrict__ output, 
    const int width, const int height, const float * const kernelX, const float * const kernelY, const int kSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= width || x >= height) return;

    float gx = 0, gy = 0;
    for (int ky = -1; ky <= 1; ++ky){
        //使用镜像边界
        int iy = y + ky;
        if (iy < 0) iy = -iy - 1;
        else if (iy >= height) iy = 2 * height - iy - 1;
        for (int kx = -1; kx <= 1; ++kx){
            int ix = x + kx;
            if (ix < 0)  ix = -ix - 1;
            else if (ix >= width) ix = 2*width - ix - 1;

            float pixel = input[iy * width + ix];
            int kIndex = (ky + 1) * kSize + (kx + 1);

            gx += (kernelX ? pixel * kernelX[kIndex] : 0);
            gy += (kernelY ? pixel * kernelY[kIndex] : 0);
        }
    }
    output[y * width + x] = ::sqrt(gx * gx + gy * gy);
}