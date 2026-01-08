#pragma once
#include <vector>
#include <string>
#include "kernel_desc.hpp"
/**
 * @brief 存放内核
 */
class filter{
public:
    filter() = default;
private:
    /**
     * @brief Construct a new filter object
     * @param ksize 核的大小
     * @param kdata 数据
     */
    filter(int ksize,std::vector<float> kdata);
public: 
    int size = 3;
    int radius = 1;
    std::vector<float> kdata;
    /**
     * @brief 自定义内核
     * @param ksize 核的大小
     * @param kdata 数据
     * @return filter 
     */
    static filter filterCustom(int ksize,std::vector<float> kdata);
    /**
     * @brief 高斯模糊核
     * @param size  核的大小
     * @param sigma 滤波器的平滑程度。sigma越大，高斯滤波器越宽，平滑效果越明显
     *              sigma的大小直接影响滤波器的权重分布，越大越模糊
     * @return filter 
     */
    static filter gaussian(int size, float sigma);
    /**
     * @brief Sobel 边缘检测（水平）
     * @return filter 
     */
    static filter sobelX();
    /**
     * @brief Sobel 边缘检测（垂直）
     * @return filter 
     */
    static filter sobelY();
    /**
     * @brief 锐化滤波器
     * @return filter 
     */
    static filter sharpen();
    /**
     * @brief 均值模糊滤波器
     * @param size 
     * @return filter 
     */
    static filter meanBlur(int size);
    /**
     * @brief 拉普拉斯算子
     * @return filter 
     */
    static filter laplacian();
    // 添加辅助函数获取内核名称
    std::string inline const static getFilterName(filter_type type) {
        switch(type) {
            case filter_type::GAUSSIAN: return "Gaussian";
            case filter_type::SOBELX: return "Sobel X";
            case filter_type::SOBELY: return "Sobel Y";
            case filter_type::SHARPEN: return "Sharpen";
            case filter_type::MEANBLUR: return "Mean Blur";
            case filter_type::LAPLACIAN: return "Laplacian";
            case filter_type::FILTERCUSTOM: return "filterCustom";
            default: return "Unknown";
        }
    }
};