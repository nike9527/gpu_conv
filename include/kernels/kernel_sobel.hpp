#pragma once
#include "core/kernel_base.hpp"
#include "filters/filter.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_memory.hpp"
#include "convolution_kernel.cuh"

class kernel_sobel final : public kernel_base {
public:
    explicit kernel_sobel(const float* d_kernel, int k): kernel(d_kernel), ksize(k) {}

    void launch(const float* in,float* out,int w, int h, cuda_stream stream,const kernel_desc& desc,const filter& filter) override;
    /**
     * @brief sobel卷积 (GPU Global)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     * @param dx x方向卷积 0不做处理
     * @param dy y方向卷积 0不做处理
     * */
    void sobelConvolutionGlobalGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy,int block_w = 16, int block_h = 16);
    /**
     * @brief sobel卷积 (GPU shared constent)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     * @param dx x方向卷积 0不做处理
     * @param dy y方向卷积 0不做处理
     * */
    void sobelConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy,int block_w = 16, int block_h = 16);
private:
    const float* kernel;
    int ksize;
};
