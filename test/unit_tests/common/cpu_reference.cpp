#include "cpu_reference.hpp"
#include <algorithm>

void cpu_convolution(const float* in,float* out,int w,int h,const std::vector<float>& k,int ks) {
    int r = ks / 2;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.f;
            for (int ky = -r; ky <= r; ++ky) {
                for (int kx = -r; kx <= r; ++kx) {
                    int ix = std::min(std::max(x + kx, 0), w - 1);
                    int iy = std::min(std::max(y + ky, 0), h - 1);
                    sum += in[iy * w + ix] *
                           k[(ky + r) * ks + (kx + r)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}
