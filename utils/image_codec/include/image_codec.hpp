#pragma once
/**
 * 简单图像加载/保存(stb_image)
 */
#include <vector>
#include <string>
/**
 * @brief BGR图片
 */
enum class RGB {B,G,R};
class Image
{
public:
    int width = 0;
    int height = 0;
    int channels = 1;
    /**
    * @brief 原始图片
    */
    std::vector<float> data;
    //灰度转换系数（RGB到灰度）
    double rgb_to_gray[3] = {0.299f, 0.587f, 0.114f};
    Image() = default;
    ~Image() = default;
    explicit  Image(int w, int h, int c=1):width(w),height(h),channels(c),data(w*h*c,0.0f){};
    // 移动构造函数
    Image(Image&& other) noexcept : width(other.width), height(other.height), channels(other.channels), data(std::move(other.data)) {
        other.width = other.height = other.channels = 0;
    }
    // 移动赋值
    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            width = other.width;
            height = other.height;
            channels = other.channels;
            data = std::move(other.data);
            other.width = other.height = other.channels = 0;
        }
        return *this;
    }
    // 禁止拷贝（使用移动语义提高性能）
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    /**
     * @brief 加载图片文件数据 
     * @param path 
     * @return Image 
     */
    static Image imageLoadFile(const std::string& path);
    /**
     * @brief 加载灰度图片文件数据 
     * @param path 
     * @return Image 
     */
    static Image imageLoadGray(const std::string& path);
    /**
     * @brief 保存图片
     * @param path 保存图片路径
     * @return true 成功
     * @return false 失败
     */
    Image imageSaveToFile(const std::string& path, RGB type);
    /**
     * @brief 保存图片
     * @param path 保存图片路径
     * @return true 成功
     * @return false 失败
     */
    bool imageSaveToFile(const std::string& path);
    /**
     * @brief 保存图片为灰度图
     * @param path 保存图片路径
     * @return true 成功
     * @return false 失败
     */
    bool imageSaveToGray(const std::string& path);
    /**
     * @brief 图片RGB分离
     * @param path 
     * @return true 成功
     * @return false 失败
     */
    bool imageSplit(const std::string& path);
};