#include "core/kernel_base.hpp"
#include "filters/filter.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_memory.hpp"
#include "convolution_kernel.cuh"
#include "kernels/kernel_sobel.hpp"
void kernel_sobel::launch(const float* in,float* out,int w, int h, cuda_stream stream,const kernel_desc& desc,const filter& filter){

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
void kernel_sobel::sobelConvolutionCPU(const float* in, float* out, const int w, const int h,const int dx, const int dy)
{
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
/**
    * @brief sobel卷积 (GPU Global)
    * @param in  输入数据
    * @param out 输入数据
    * @param w   高度  
    * @param h   宽度
    * @param dx x方向卷积 0不做处理
    * @param dy y方向卷积 0不做处理
    * */
void kernel_sobel::sobelConvolutionGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy,int block_w, int block_h){
    filter kernelX = filter::sobelX();
    filter kernelY = filter::sobelY();
    int kSize = kernelX.size;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernelX(kSize*kSize);
    cuda_memory<float> d_kernelY(kSize*kSize);
    d_input.copy_from_host(in, w * h);
    d_kernelX.copy_from_host(kernelX.kdata.data(),kSize*kSize);
    d_kernelY.copy_from_host(kernelY.kdata.data(),kSize*kSize);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    // start.record();
    sobelConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernelX.data(), d_kernelY.data(), kSize);
    // stop.record(); 
    // cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
} 
/**
    * @brief sobel卷积 (GPU shared constent)
    * @param in  输入数据
    * @param out 输入数据
    * @param w   高度  
    * @param h   宽度
    * @param dx x方向卷积 0不做处理
    * @param dy y方向卷积 0不做处理
    * */
void kernel_sobel::sobelConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy,int block_w, int block_h){
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    dim3 block(block_w,block_h);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    // start.record();
    size_t shraedXSize = block.y * (block.x + 2) * sizeof(float);
    size_t shraedYSize = (block.y + 2) * block.x * sizeof(float);
    sobelXConvolutionWithShared<<<grid, block, shraedXSize>>>(d_input.data(), d_output.data(), w, h);
    sobelYConvolutionWithShared<<<grid, block, shraedYSize>>>(d_input.data(), d_output.data(), w, h);
    // stop.record(); 
    // cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out,w * h);
    return;
} 
