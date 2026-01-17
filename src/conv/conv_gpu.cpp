#include "conv/conv_gpu.hpp"
#include "filters/kernel_desc.hpp"
#include "kernels/kernels.cuh"
namespace gpu_conv
{
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
    void conv2dKernel(const float *in, float *out, const int width, const int height, mem_type type, const filter &filterObj)
    {
        filter_pipeline pipe(width, height, cudaStreamNonBlocking);
        launchFilter(pipe, in, out, type, filterObj);
        pipe.stream.synchronize();
    }
    /**
     * @brief 高斯卷积(cpu omp)
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param kSize  核大小
     */
    void gaussianBlur(const float *in, float *out, const int width, const int height, mem_type type, int size, float sigma)
    {
        filter_pipeline pipe(width, height, cudaStreamNonBlocking);
        launchGaussianBlur(pipe, in, out, type, size, sigma);
        pipe.stream.synchronize();
    }
    /**
     * @brief 拉普拉斯算子(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     */
    void laplacian(const float *in, float *out, const int width, const int height, mem_type type)
    {
        filter_pipeline pipe(width, height, cudaStreamNonBlocking);
        launchLaplacian(pipe, in, out, type);
        pipe.stream.synchronize();
    }
    /**
     * @brief  均值模糊(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     * @param kSize  核大小
     */
    void meanBlur(const float *in, float *out, const int width, const int height, mem_type type, int size)
    {
        filter_pipeline pipe(width, height, cudaStreamNonBlocking);
        launchMeanBlur(pipe, in, out, type, size);
        pipe.stream.synchronize();
    }
    /**
     * @brief  锐化滤波器(cpu omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度
     * @param h   宽度
     */
    void sharpen(const float *in, float *out, const int width, const int height, mem_type type)
    {
        filter_pipeline pipe(width, height, cudaStreamNonBlocking);
        launchsharpen(pipe, in, out, type);
        pipe.stream.synchronize();
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
    void sobel(const float *in, float *out, const int width, const int height, mem_type type)
    {
        filter_pipeline pipe(width, height, cudaStreamNonBlocking);
        launchSobel(pipe, in, out, type, 3);
        pipe.stream.synchronize();
    }
    /**
     * @brief
     *
     * @param pipe
     * @param in
     * @param out
     * @param width
     * @param height
     * @param filter_type
     * @param block_w
     * @param block_h
     */
    void launchFilterAsync(filter_pipeline &pipe, const float *in, float *out, const int width, const int height, const filter_type filter_type, int block_w, int block_h)
    {
        filter obj = filter::getFilterObj(filter_type);
        launchFilter(pipe, in, out, mem_type::SHAREDCONST, obj);
    }
#include "pipeline/gl_frame_slot.hpp"
    void launchGaussianRGBA(const float *in)
    {
        gl_frame_slot gl_pipe;
        gl_pipe.pbo = std::make_unique<GLPBO>(800, 800);
        gl_pipe.pbo->map(gl_pipe.stream);
        // gaussianRGBAGPU(gl_pipe, in, 800, 800);
        cudaMemsetAsync(gl_pipe.pbo->device_ptr(), 0, gl_pipe.pbo->size_bytes(), gl_pipe.stream.get());
        gl_pipe.stream.synchronize();
        gl_pipe.pbo->unmap(gl_pipe.stream.get());
    }
}