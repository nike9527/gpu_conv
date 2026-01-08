#include "core/kernel_base.hpp"
#include "filters/filter.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_memory.hpp"
#include "convolution_kernel.cuh"
#include "kernels/kernel_filter.hpp"

void kernel_filter::launch(const float* in,float* out,int w, int h, cuda_stream stream, const kernel_desc& desc,const filter& filter) {
    if(desc.dev_type == dev_type::DEVCPU){
        conv2dCpuOmp(in,out,w,h,filter);
    }else if(desc.dev_type == dev_type::DEVGPU && desc.mem_type == mem_type::GLOBAL){
        conv2dGlobalGPU(in,out,w,h,filter);
    }else if(desc.dev_type == dev_type::DEVGPU && desc.mem_type == mem_type::SHAREDCONST){
        conv2dWithSharedGPU(in,out,w,h,filter);
    }
}
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
void kernel_filter::conv2dCpuOmp(const float* in, float* out, const int w, const int h, const filter& filter){
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
    * @brief 自定义卷积(GPU Global)
    * @param in  输入数据
    * @param out 输入数据
    * @param w   高度  
    * @param h   宽度
    * @param kSize 核大小
    * @param kernel 核
    */
void kernel_filter::conv2dGlobalGPU(const float* in, float* out, const int w, const int h, const filter& filter,int block_w, int block_h){
    // cuda_event start, stop;
    int kSize = filter.size;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernel(kSize*kSize);
    d_input.copy_from_host(in, w * h);
    d_kernel.copy_from_host(filter.kdata.data(),kSize*kSize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    // start.record();
    conv2dGlobalKernel<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
    // stop.record();
    // cudaEventSynchronize(stop);// 等待事件完成 
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
/**
    * @brief 自定义卷积(GPU shared constent)
    * @param in  输入数据
    * @param out 输入数据
    * @param w   高度  
    * @param h   宽度
    * @param kSize 核大小
    * @param kernel 核
    */
void kernel_filter::conv2dWithSharedGPU(const float* in, float* out, const int w, const int h, const filter& filter, int block_w, int block_h){
    int kSize = filter.size;
    int r = kSize / 2;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    CUDA_CHECK(cudaMemcpyToSymbol(constkernel, filter.kdata.data(), kSize*kSize*sizeof(float), 0, cudaMemcpyHostToDevice));
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    int shraedSize = (block_w + 2*r) * (block_h + 2*r) * sizeof(float);
    // start.record();
    conv2dGlobalKernelWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
    // stop.record(); 
    // cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
