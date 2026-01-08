#pragma once
#include "core/kernel_base.hpp"
class kernel_filter final : public kernel_base {
public:
    explicit kernel_filter(const float* d_kernel, int k): kernel(d_kernel), ksize(k) {}

    void launch(const float* in,float* out,int w, int h, cuda_stream stream, const kernel_desc& desc,const filter& filter) override;
    /**
     * @brief 自定义卷积(GPU Global)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     * @param kSize 核大小
     * @param kernel 核
     */
    void convolutionGPU(const float* in, float* out, const int w, const int h, const filter& filter,int block_w = 16, int block_h = 16);
    /**
     * @brief 自定义卷积(GPU shared constent)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     * @param kSize 核大小
     * @param kernel 核
     */
    void convolutionWithSharedGPU(const float* in, float* out, const int w, const int h, const filter& filter,int block_w = 16, int block_h = 16);
private:
    const float* kernel;
    int ksize;
};
