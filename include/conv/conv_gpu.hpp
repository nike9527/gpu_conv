#include "filters/filter.hpp"
namespace gpu_conv {
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
    void convolutionGPU(const float* in, float* out, const int w, const int h, const filter& filter);
     /**
     * @brief 高斯卷积(GPU )
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param kSize  核大小
     */
    void gaussianConvolutionGPU(const float* in, float* out, const int w, const int h, const filter& filter);
    /**
     * @brief 拉普拉斯算子(GPU omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     */
    void laplacianConvolutionGPU(const float* in, float* out, const int w, const int h,const filter& filter);
    /**
     * @brief  均值模糊(GPU omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     * @param kSize  核大小
     */
    void meanBlurConvolutionGPU(const float* in, float* out, const int w, const int h,const filter& filter);
    /**
     * @brief  锐化滤波器(GPU omp)
     * @param in  输入数据
     * @param out 输入数据
     * @param w   高度  
     * @param h   宽度
     */
    void sharpenConvolutionGPU(const float* in, float* out, const int w, const int h,const filter& filter);
   /**
     * @brief sobel卷积(GPU omp)
     * @param in  输入数据
     * @param out   输出数据
     * @param w  宽度
     * @param h  高度
     * @param dx x方向卷积
     * @param dy y方向卷积
     */
    void sobelConvolutionGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy);
};