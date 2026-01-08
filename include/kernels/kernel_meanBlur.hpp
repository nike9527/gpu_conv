#pragma once
#include "core/kernel_base.hpp"

class kernel_meanBlur final : public kernel_base {
public:
    explicit kernel_meanBlur(const float* d_kernel, int k): kernel(d_kernel), ksize(k) {}

    void launch(const float* in,float* out,int w, int h, cuda_stream stream, const kernel_desc& desc,const filter& filter) override;

    /**
     * @brief 均值模糊 (GPU Global)
     * @param input 输入数据
     * @param output 输出数据
     * @param width 宽度
     * @param height 高度
     * @param kernel 内核
     * @param ksize 内核大小
     */
    void meanBlurConvolutionGlobalGPU(const float* in, float* out, const int w, const int h,const filter& filter,int block_w = 16, int block_h = 16);
    /**
     * @brief 均值模糊 (GPU shared constent)
     * @param input 输入数据
     * @param output 输出数据
     * @param width 宽度
     * @param height 高度
     * @param kernel 内核
     * @param ksize 内核大小
     */
    void meanBlurConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,const filter& filter,int block_w = 16, int block_h = 16);
private:
    const float* kernel;
    int ksize;
};
