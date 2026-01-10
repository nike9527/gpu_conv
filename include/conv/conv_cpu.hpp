#pragma once
#include "filters/filter.hpp"
namespace cpu_conv
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
  void conv2dKernel(const float *in, float *out, const int w, const int h, const filter &filter);
  /**
   * @brief 高斯卷积(cpu omp)
   * @param in  输入数据
   * @param out   输出数据
   * @param w  宽度
   * @param h  高度
   * @param kSize  核大小
   */
  void gaussianBlur(const float *in, float *out, const int w, const int h, const int size, const float sigma);
  void gaussianBlur1D(const float *in, float *out, const int w, const int h, const int size, const float sigma);
  void gaussianBlur2D(const float *in, float *out, const int w, const int h, const int size, const float sigma);
  /**
   * @brief 拉普拉斯算子(cpu omp)
   * @param in  输入数据
   * @param out 输入数据
   * @param w   高度
   * @param h   宽度
   */
  void laplacian(const float *in, float *out, const int w, const int h);
  /**
   * @brief  均值模糊(cpu omp)
   * @param in  输入数据
   * @param out 输入数据
   * @param w   高度
   * @param h   宽度
   * @param kSize  核大小
   */
  void meanBlur(const float *in, float *out, const int w, const int h, int size);
  /**
   * @brief  锐化滤波器(cpu omp)
   * @param in  输入数据
   * @param out 输入数据
   * @param w   高度
   * @param h   宽度
   */
  void sharpen(const float *in, float *out, const int w, const int h);
  /**
   * @brief sobel卷积(cpu omp)
   * @param in  输入数据
   * @param out   输出数据
   * @param w  宽度
   * @param h  高度
   * @param dx x方向卷积
   * @param dy y方向卷积
   */
  void sobel(const float *in, float *out, const int width, const int height);
};