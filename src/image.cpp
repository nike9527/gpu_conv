/**
 * 简单图像加载/保存(stb_image)
 */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "image.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"
#include <fstream>
#include <iostream>
Image::Image(){}
Image::Image(int w, int h,int c):width(w),height(h),channels(c),data(w*h*c,0.0f){}
Image::~Image(){}
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

Image Image::imageLoadGray(const std::string& path){
    int width,height,channels;
    unsigned char* stImg = stbi_load(path.c_str(), &width, &height, &channels, 1);
    Image img(width,height);
    for (int i = 0; i < width * height; ++i)
        img.data[i] = stImg[i] / 255.0f;
    stbi_image_free(stImg); 
    return img;
}


Image Image::imageSaveToFile(const std::string& path, RGB type){
    return Image();
}

bool Image::imageSaveToFile(const std::string& path){
    std::vector<unsigned char> out(width * height);
    for (int i = 0; i < data.size(); ++i)
        out[i] = static_cast<unsigned char>(std::min(1.0f, std::max(0.0f, data[i])) * 255);
    stbi_write_png(path.c_str(), width, height, 1, out.data(), width);
    return true;
}

bool Image::imageSaveToGray(const std::string& path){
    imageSaveToFile(path);
    return true;
}
 bool Image::imageSplit(const std::string& path){
    return true;
 }