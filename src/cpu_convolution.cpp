/**
 * CPU baseline + OpenMP
 */
#include "cpu_convolution.hpp"
#include <algorithm>
#include <omp.h>

void convolution(const float* in, float* out, int w, int h, const float* kernel, int ksize) {
    int r = ksize / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int ky = -r; ky <= r; ky++) {
                for (int kx = -r; kx <= r; kx++) {
                    int ix = std::min(std::max(x + kx, 0), w - 1);
                    int iy = std::min(std::max(y + ky, 0), h - 1);
                    sum += in[iy * w + ix] * kernel[(ky + r) * ksize + (kx + r)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}

void gaussianConvolution(const float* in, float* out, int w, int h, const float* kernel, int kSize){
    int radius = kSize / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
                // 使用镜像边界
                int iy = y + ky;
                 // 镜像处理
                // if (iy < 0) iy = -iy;
                // else if (iy >= h) iy = h - 1;
                if (iy < 0) iy = -iy - 1; // 镜像：0 → -1 → 0, -1 → -2 → 1
                else if (iy >= h) iy = 2 * h - iy - 1; // 镜像：h → h-1, h+1 → h-2
                for (int kx = -radius; kx <= radius; ++kx) {
                    // 使用镜像边界
                    int ix = x + kx;
                     // 镜像处理
                    // if (ix < 0) ix = -ix;
                    // else if (ix >= w) ix = w - 1;
                    if (ix < 0)  ix = -ix - 1;
                    else if (ix >= w) ix = 2*w - ix - 1;
                    sum += in[iy * w + ix] * kernel[(ky + radius) * kSize + (kx + radius)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}

