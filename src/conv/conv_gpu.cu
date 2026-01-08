#include "conv/conv_gpu.hpp"
#include "kernels/kernel_filter.hpp"
#include "kernels/kernel_gaussian.hpp"
#include "kernels/kernel_laplacian.hpp"
#include "kernels/kernel_meanBlur.hpp"
#include "kernels/kernel_sharpen.hpp"
#include "kernels/kernel_sobel.hpp"
namespace gpu_conv
{
  /**
   * @brief 自自定义卷积
   *
   * @param in  输入数据
   * @param out   输出数据
   * @param w  宽度
   * @param h  高度
   */
  void convolutionGPU(const float *in, float *out, const int w, const int h, int block_w, int block_h, mem_type type, const filter &filter)
  {
    // cuda_event start, stop;
    int kSize = filter.size;
    int r = kSize / 2;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    dim3 block(block_w, block_h);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    if (type == mem_type::GLOBAL)
    {
      cuda_memory<float> d_kernel(kSize * kSize);
      d_kernel.copy_from_host(filter.kdata.data(), kSize * kSize);
      // start.record()
      conv2dGlobalKernel<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
      CUDA_CHECK(cudaMemcpyToSymbol(constkernel, filter.kdata.data(), kSize * kSize * sizeof(float), 0, cudaMemcpyHostToDevice));
      int shraedSize = (block_w + 2 * r) * (block_h + 2 * r) * sizeof(float);
      // start.record()
      conv2dGlobalKernelWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out, w * h);
    return;
  }
  /**
   * @brief 高斯卷积(GPU )
   * @param in  输入数据
   * @param out   输出数据
   * @param w  宽度
   * @param h  高度
   * @param kSize  核大小
   */
  void gaussianConvolutionGPU(const float *in, float *out, const int w, const int h, int block_w, int block_h, mem_type type, const filter &filter)
  {
    int ksize = filter.size;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);

    dim3 block(block_w, block_h);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    if (type == mem_type::GLOBAL)
    {
      cuda_memory<float> d_kernel(ksize * ksize);
      d_kernel.copy_from_host(filter.kdata.data(), ksize * ksize);
      // start.record();
      gaussianConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), ksize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
      CUDA_CHECK(cudaMemcpyToSymbol(constkernel, filter.kdata.data(), ksize * ksize * sizeof(float), 0, cudaMemcpyHostToDevice));
      int shraedSize = (block_w + 2 * filter.radius) * (block_h + 2 * filter.radius) * sizeof(float);
      // start.record();
      gaussianConvolutionWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, ksize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }

    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out, w * h);
    return;
  }
  /**
   * @brief 拉普拉斯算子(GPU omp)
   * @param in  输入数据
   * @param out 输入数据
   * @param w   高度
   * @param h   宽度
   */
  void laplacianConvolutionGPU(const float *in, float *out, const int w, const int h, int block_w, int block_h, mem_type type, const filter &filter)
  {
    int kSize = filter.size;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    dim3 block(block_w, block_h);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    if (type == mem_type::GLOBAL)
    {
      cuda_memory<float> d_kernel(kSize * kSize);
      d_kernel.copy_from_host(filter.kdata.data(), kSize * kSize);
      // start.record();
      laplacianConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
      CUDA_CHECK(cudaMemcpyToSymbol(constkernel, filter.kdata.data(), kSize * kSize * sizeof(float), 0, cudaMemcpyHostToDevice));
      int shraedSize = (block_w + 2 * filter.radius) * (block_h + 2 * filter.radius) * sizeof(float);
      // start.record();
      laplacianConvolutionWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }

    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out, w * h);
    return;
  }
  /**
   * @brief  均值模糊(GPU omp)
   * @param in  输入数据
   * @param out 输入数据
   * @param w   高度
   * @param h   宽度
   * @param kSize  核大小
   */
  void meanBlurConvolutionGPU(const float *in, float *out, const int w, const int h, int block_w, int block_h, mem_type type, const filter &filter)
  {
    int ksize = filter.size;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    dim3 block(block_w, block_h);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    if (type == mem_type::GLOBAL)
    {
      cuda_memory<float> d_kernel(ksize * ksize);
      d_kernel.copy_from_host(filter.kdata.data(), ksize * ksize);
      // start.record();
      meanBlurConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), ksize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
      CUDA_CHECK(cudaMemcpyToSymbol(constkernel, filter.kdata.data(), ksize * ksize * sizeof(float), 0, cudaMemcpyHostToDevice));
      int shraedSize = (block_w + 2 * filter.radius) * (block_h + 2 * filter.radius) * sizeof(float);
      // start.record();
      meanBlurConvolutionWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, ksize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }

    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out, w * h);
    return;
  }
  /**
   * @brief  锐化滤波器(GPU omp)
   * @param in  输入数据
   * @param out 输入数据
   * @param w   高度
   * @param h   宽度
   */
  void sharpenConvolutionGPU(const float *in, float *out, const int w, const int h, int block_w, int block_h, mem_type type, const filter &filter)
  {
    int kSize = filter.size;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    dim3 block(block_w, block_h);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    if (type == mem_type::GLOBAL)
    {
      cuda_memory<float> d_kernel(kSize * kSize);
      d_kernel.copy_from_host(filter.kdata.data(), kSize * kSize);
      // start.record();
      sharpenConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernel.data(), kSize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }
    else if (type == mem_type::SHAREDCONST)
    {
      CUDA_CHECK(cudaMemcpyToSymbol(constkernel, filter.kdata.data(), kSize * kSize * sizeof(float), 0, cudaMemcpyHostToDevice));
      int shraedSize = (block_w + 2 * filter.radius) * (block_h + 2 * filter.radius) * sizeof(float);
      // start.record();
      sharpenConvolutionWithShared<<<grid, block, shraedSize>>>(d_input.data(), d_output.data(), w, h, kSize);
      // stop.record();
      // cudaEventSynchronize(stop);// 等待事件完成
    }

    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out, w * h);
    return;
  }
  /**
   * @brief sobel卷积(GPU omp)
   * @param in  输入数据
   * @param out   输出数据
   * @param w  宽度
   * @param h  高度
   * @param dx x方向卷积
   * @param dy y方向卷积
   */
  void sobelConvolutionGPU(const float *in, float *out, const int w, const int h, const int dx, const int dy, int block_w, int block_h, mem_type type)
  {
    filter kernelX = filter::sobelX();
    filter kernelY = filter::sobelY();
    int kSize = kernelX.size;
    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    cuda_memory<float> d_kernelX(kSize * kSize);
    cuda_memory<float> d_kernelY(kSize * kSize);
    d_input.copy_from_host(in, w * h);
    d_kernelX.copy_from_host(kernelX.kdata.data(), kSize * kSize);
    d_kernelY.copy_from_host(kernelY.kdata.data(), kSize * kSize);
    dim3 block(block_w, block_h);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    // start.record();
    sobelConvolution<<<grid, block>>>(d_input.data(), d_output.data(), w, h, d_kernelX.data(), d_kernelY.data(), kSize);
    // stop.record();
    // cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out, w * h);
    return;

    // cuda_event start, stop;
    cuda_memory<float> d_input(w * h);
    cuda_memory<float> d_output(w * h);
    d_input.copy_from_host(in, w * h);
    dim3 block(block_w, block_h);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    // start.record();
    size_t shraedXSize = block.y * (block.x + 2) * sizeof(float);
    size_t shraedYSize = (block.y + 2) * block.x * sizeof(float);
    sobelXConvolutionWithShared<<<grid, block, shraedXSize>>>(d_input.data(), d_output.data(), w, h);
    sobelYConvolutionWithShared<<<grid, block, shraedYSize>>>(d_input.data(), d_output.data(), w, h);
    // stop.record();
    // cudaEventSynchronize(stop);// 等待事件完成
    CHECK_KERNEL_ERROR();
    // std::cout << "GPU time: " << start.elapsed_ms(stop) << " ms\n";
    d_output.copy_to_host_async(out, w * h);
    return;
  }
};