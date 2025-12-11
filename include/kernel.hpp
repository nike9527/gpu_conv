#pragma once
#include <vector>
/**
 * @brief 存放内核
 */
class Kernel{
private:
    /**
     * @brief Construct a new Kernel object
     * @param ksize 核的大小
     * @param kdata 数据
     */
    Kernel(int ksize,std::vector<float> kdata);
public: 
    int size = 3;
    bool isConMen = true;
    std::vector<float> kdata;
    /**
     * @brief 自定义内核
     * @param ksize 核的大小
     * @param kdata 数据
     * @return Kernel 
     */
    static Kernel filterKernel(int ksize,std::vector<float> kdata);
    /**
     * @brief 高斯模糊核
     * @param size  核的大小
     * @param sigma 滤波器的平滑程度。sigma越大，高斯滤波器越宽，平滑效果越明显
     *              sigma的大小直接影响滤波器的权重分布，越大越模糊
     * @return Kernel 
     */
    static Kernel gaussian(int size, float sigma);
    /**
     * @brief Sobel 边缘检测（水平）
     * @return Kernel 
     */
    static Kernel sobelX();
    /**
     * @brief Sobel 边缘检测（垂直）
     * @return Kernel 
     */
    static Kernel sobelY();
    /**
     * @brief 锐化滤波器
     * @return Kernel 
     */
    static Kernel sharpen();
    /**
     * @brief 均值模糊滤波器
     * @param size 
     * @return Kernel 
     */
    static Kernel meanBlur(int size);
    /**
     * @brief 拉普拉斯算子
     * @return Kernel 
     */
    static Kernel laplacian();
};