#include "kernel.hpp"
#include <cmath>
#include <iostream>
/**
 * @brief Construct a new Kernel object
 * @param size 
 * @param kdata 
 */
Kernel::Kernel(int ksize, std::vector<float> kdata):size(ksize),kdata(kdata){}

Kernel Kernel::filterKernel(int ksize,std::vector<float> kdata){
  return Kernel(ksize*ksize,kdata);
}
/**
 * @brief 高斯模糊核
 * @param size  核的大小
 * @param sigma 滤波器的平滑程度。sigma越大，高斯滤波器越宽，平滑效果越明显
 *              sigma的大小直接影响滤波器的权重分布，越大越模糊 
 *              对于n×n 的核，建议sigma取值 ≈ n/3 到 n/2
 * @return Kernel 
 */
Kernel Kernel::gaussian(int size, float sigma) {
    Kernel k(size,std::vector<float>(size*size, 0.0f));
    std::vector<float> data(size*size, 0.0f);
    int r = size/2;
    float sum = 0.0f;
    #pragma omp parallel for
    for (int y=-r;y<=r;y++){
        for (int x=-r;x<=r;x++){
            float v = expf(-(x*x+y*y)/(2*sigma*sigma));
            k.kdata[(y+r)*size + (x+r)] = v;
            sum += v;
        }
    }
    #pragma omp parallel for
    for (int i=0;i<k.kdata.size();i++){
        k.kdata[i] /= sum;
    }
    return k;
}
/**
 * @brief Sobel 边缘检测（水平）
 * @return Kernel 
 */
Kernel Kernel::sobelX(){
    return Kernel(3,{-1.0f, 0.0f, 1.0f,-2.0f, 0.0f, 2.0f,-1.0f, 0.0f, 1.0f});
}
/**
 * @brief Sobel 边缘检测（垂直）
 * @return Kernel 
 */
Kernel Kernel::sobelY(){
    return Kernel(3,{ -1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f });
}
/**
 * @brief 锐化滤波器
 * @return Kernel 
 */
Kernel Kernel::sharpen(){
    return Kernel(3,{ 0.0f, -1.0f, 0.0f, -1.0f, 5.0f, -1.0f, 0.0f, -1.0f, 0.0f});
}
/**
 * @brief 均值模糊滤波器
 * @param size 
 * @return Kernel 
 */
Kernel Kernel::meanBlur(int size){
    return Kernel(size,std::vector<float>(size * size, 1.0f / (size * size)));
}
/**
 * @brief 拉普拉斯算子
 * @return Kernel 
 */
Kernel Kernel::laplacian(){
   return Kernel(3,{ 0.0f,-1.0f, 0.0f,-1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f });
}