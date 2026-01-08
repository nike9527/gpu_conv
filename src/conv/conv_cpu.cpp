#pragma once
#include "conv/conv_cpu.hpp"
namespace cpu_conv {
    /**
     * @brief 自自定义卷积
     * 
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param kernel 内核
     * @param kSize  核大小
     */
     void convolutionCpu(const float* in, float* out, const int w, const int h, const filter& filter){
          int r = filter.size / 2;
          #pragma omp parallel for
          for (int y = 0; y < h; y++) {
              for (int x = 0; x < w; x++) {
                  float sum = 0.0f;
                  for (int ky = -r; ky <= r; ky++) {
                      for (int kx = -r; kx <= r; kx++) {
                          int ix = std::min(std::max(x + kx, 0), w - 1);
                          int iy = std::min(std::max(y + ky, 0), h - 1);
                          sum += in[iy * w + ix] * filter.kdata[(ky + r) * filter.size + (kx + r)];
                      }
                  }
                  out[y * w + x] = sum;
              }
          }
     }
     /**
     * @brief 高斯卷积(cpu omp)
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param kSize  核大小
     */
    void gaussianConvolutionCPU(const float* in, float* out, const int w, const int h, const filter& filter){
          int radius = filter.radius;
          int ksize = filter.size;
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
                          sum += in[iy * w + ix] * filter.kdata[(ky + radius) * ksize + (kx + radius)];
                      }
                  }
                  out[y * w + x] = sum;
              }
          }
     }
    /**
     * @brief 拉普拉斯算子(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     */
    void laplacianConvolutionCPU(const float* in, float* out, const int w, const int h,const filter& filter){
      int kSize = filter.size;
      int radius = filter.radius;
      #pragma omp parallel for
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
                // 使用镜像边界
                int iy = y + ky;
                if (iy < 0) iy = -iy - 1; 
                else if (iy >= h) iy = 2 * h - iy - 1;
                for (int kx = -radius; kx <= radius; ++kx) {
                    // 使用镜像边界
                    int ix = x + kx;
                    if (ix < 0)  ix = -ix - 1;
                    else if (ix >= w) ix = 2*w - ix - 1;
                    sum += in[iy * w + ix] * filter.kdata[(ky + radius) * kSize + (kx + radius)];
                }
            }
            out[y * w + x] = sum;
         }
      }
    }
    /**
     * @brief  均值模糊(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     * @param kSize  核大小
     */
    void meanBlurConvolutionCPU(const float* in, float* out, const int w, const int h,const filter& filter){
          int ksize = filter.size;
          int radius = filter.radius;
          #pragma omp parallel for
          for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                  float sum = 0.0f;
                  for (int ky = -radius; ky <= radius; ++ky) {
                      // 使用镜像边界
                      int iy = y + ky;
                      if (iy < 0) iy = -iy - 1; 
                      else if (iy >= h) iy = 2 * h - iy - 1;
                      for (int kx = -radius; kx <= radius; ++kx) {
                          // 使用镜像边界
                          int ix = x + kx;
                          if (ix < 0)  ix = -ix - 1;
                          else if (ix >= w) ix = 2*w - ix - 1;
                          sum += in[iy * w + ix] * filter.kdata[(ky + radius) * ksize + (kx + radius)];
                      }
                  }
                  out[y * w + x] = sum;
              }
          }
     }
    /**
     * @brief  锐化滤波器(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     */
    void sharpenConvolutionCPU(const float* in, float* out, const int w, const int h,const filter& filter){
          int kSize = filter.size;
          int radius = filter.radius;
          #pragma omp parallel for
          for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                  float sum = 0.0f;
                  for (int ky = -radius; ky <= radius; ++ky) {
                      // 使用镜像边界
                      int iy = y + ky;
                      if (iy < 0) iy = -iy - 1; 
                      else if (iy >= h) iy = 2 * h - iy - 1;
                      for (int kx = -radius; kx <= radius; ++kx) {
                          // 使用镜像边界
                          int ix = x + kx;
                          if (ix < 0)  ix = -ix - 1;
                          else if (ix >= w) ix = 2*w - ix - 1;
                          sum += in[iy * w + ix] * filter.kdata[(ky + radius) * kSize + (kx + radius)];
                      }
                  }
                  out[y * w + x] = sum;
              }
          }
     }
   /**
     * @brief sobel卷积(cpu omp)
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param dx x方向卷积
     * @param dy y方向卷积
     */
    void sobelConvolutionCPU(const float* in, float* out, const int w, const int h,const int dx, const int dy){
            filter kernelX = filter::sobelX();
            filter kernelY = filter::sobelY();
            int kSize = kernelX.size;
            int radius = kernelX.size /  2;
            #pragma omp parallel for
            for (int y = 0; y < h; ++y){
                for (int x = 0; x < w; ++x){
                    float gx = 0, gy = 0;
                    for (int ky = -radius; ky <= radius; ++ky){
                        //使用镜像边界
                        int iy = y + ky;
                        if (iy < 0) iy = -iy - 1;
                        else if (iy >= h) iy = 2 * h - iy - 1;
                        for (int kx = -radius; kx <= radius; ++kx){
                            int ix = x + kx;
                            if (ix < 0)  ix = -ix - 1;
                            else if (ix >= w) ix = 2*w - ix - 1;

                            float pixel = in[iy * w + ix];
                            int kIndex = (ky + radius) * kSize + (kx + radius);

                            gx += (dx ? pixel * kernelX.kdata[kIndex] : 0);
                            gy += (dy ? pixel * kernelY.kdata[kIndex] : 0);
                        }
                    }
                    out[y * w + x] = ::sqrt(gx * gx + gy * gy);
                }
            }
     }
}