#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"
#include "image_codec.hpp"
#include <fstream>
#include <iostream>

/**
 * @brief 加载图片文件数据 
 * @param path 
 * @return Image 
 */
Image Image::imageLoadFile(const std::string& path){
    int width,height,channels;
    unsigned char* rawData = stbi_load(path.c_str(), &width, &height,&channels, 0);  // 0表示保持原始通道数
    if (channels < 3) {
        stbi_image_free(rawData);
    }
    Image img(width,height,channels);
    int size = width * height * channels;
    for (size_t i = 0; i < size; ++i) {
        img.data[i] = rawData[i] / 255.0f;  
    }
    stbi_image_free(rawData);
    return img;
}
/**
 * @brief 加载灰度图片文件数据 
 * @param path 
 * @return Image 
 */
Image Image::imageLoadGray(const std::string& path){
    int width,height,channels;
    unsigned char* stImg = stbi_load(path.c_str(), &width, &height, &channels, 1);
    Image img(width,height);
    for (int i = 0; i < width * height; ++i)
        img.data[i] = stImg[i] / 255.0f;
    stbi_image_free(stImg); 
    return img;
}
/**
 * @brief 保存图片
 * @param path 保存图片路径
 * @return true 成功
 * @return false 失败
 */
Image Image::imageSaveToFile(const std::string& path, RGB type){
    return Image();
}
/**
 * @brief 保存图片
 * @param path 保存图片路径
 * @return true 成功
 * @return false 失败
 */
bool Image::imageSaveToFile(const std::string& path){
    std::vector<unsigned char> out(width * height);
    for (int i = 0; i < data.size(); ++i)
        out[i] = static_cast<unsigned char>(std::min(1.0f, std::max(0.0f, data[i])) * 255);
    stbi_write_png(path.c_str(), width, height, 1, out.data(), width);
    return true;
}
/**
 * @brief 保存图片为灰度图
 * @param path 保存图片路径
 * @return true 成功
 * @return false 失败
 */
bool Image::imageSaveToGray(const std::string& path){
    imageSaveToFile(path);
    return true;
}
/**
 * @brief 图片RGB分离
 * @param path 
 * @return true 成功
 * @return false 失败
 */
bool Image::imageSplit(const std::string& path){
    return true;
}