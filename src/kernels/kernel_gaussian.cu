#include "core/kernel_base.hpp"
#include "filters/filter.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_memory.hpp"
#include "convolution_kernel.cuh"
#include "kernels/kernel_gaussian.hpp"

/**
 * @brief 
 * 
 * @param in 
 * @param out 
 * @param w 
 * @param h 
 * @param stream 
 * @param desc 
 */
void kernel_gaussian::launch(const float* in,float* out,int w, int h, cuda_stream stream, const kernel_desc& desc,const filter& filter) {
    if(desc.dev_type == dev_type::DEVCPU){
        gaussianConvolutionCPU(in,out,w,h,filter);
    }else if(desc.dev_type == dev_type::DEVGPU && desc.mem_type == mem_type::GLOBAL){
        gaussianConvolutionGPU(in,out,w,h,filter);
    }else if(desc.dev_type == dev_type::DEVGPU && desc.mem_type == mem_type::SHAREDCONST){
        gaussianConvolutionWithSharedGPU(in,out,w,h,filter);
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
void kernel_gaussian::gaussianConvolutionCPU(const float* in, float* out, const int w, const int h, const filter& filter){
    int radius = filter.radius;
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
    * @brief 高斯卷积(GPU Global)
    * @param in  输入数据
    * @param out 输入数据
    * @param w   高度  
    * @param h   宽度
    * @param kSize 积核大小
    */
void kernel_gaussian::gaussianConvolutionGPU(const float* in, float* out, const int w, const int h, const filter& filter,int block_w, int block_h) {
    int ksize = filter.size;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernel(ksize*ksize);
    d_input.copy_from_host(in, w * h);
    d_kernel.copy_from_host(filter.kdata.data(),ksize*ksize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    // start.record();
    gaussianConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), ksize);
    // stop.record(); 
    // cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}
/**
    * @brief 高斯卷积(GPU shared constent)
    * @param in  输入数据
    * @param out 输入数据
    * @param w   高度  
    * @param h   宽度
    * @param kSize 积核大小
    */
void kernel_gaussian::gaussianConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h, const filter& filter,int block_w, int block_h) {
    int ksize = filter.size;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    CUDA_CHECK(cudaMemcpyToSymbol(constkernel, filter.kdata.data(), ksize*ksize*sizeof(float), 0, cudaMemcpyHostToDevice));
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    int shraedSize = (block_w + 2*filter.radius) * (block_h + 2*filter.radius) * sizeof(float);
    // start.record();
    gaussianConvolutionWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, ksize);
    // stop.record(); 
    // cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
}

