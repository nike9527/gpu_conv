/**
 * 卷积统一入口（对外 API）
 */
#include "convolution_gpu.hpp"
#include "convolution_cpu.hpp"
#include "kernel.hpp"
#include "filter.hpp"
#include "image_viewer.hpp"
#include <iostream>
#include <chrono>
namespace gconv {

/**
 * @brief 自定义卷积函数
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param k 内核矩阵
 * @param backend 
 * @return true 
 * @return false 
 */
bool convolve(){
    std::string srcPaht="D:/C++/gpu_conv_lib_cmake/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv_lib_cmake/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv_lib_cmake/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    // Kernel kernel = Kernel::gaussian(5, 5.0f);
    // Kernel kernel = Kernel::sharpen();
    // Kernel kernel = Kernel::meanBlur(9);
     Kernel kernel = Kernel::laplacian();
    //================CPU进行卷积计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    conv2dCpuOmp(imgData.data.data(),out.data.data(),imgData.width,imgData.height, kernel.size, kernel.kdata.data());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行卷积计算-全局内存=====================
    conv2dGlobalGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height, kernel.size,kernel.kdata.data());
    out.imageSaveToFile(destPaht3);
    //================GPU进行卷积计算-共享内存=====================
    conv2dGlobalGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height, kernel.size,kernel.kdata.data());
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
    return true;
}
/**
 * @brief 高斯滤波入口（API）
 * @return true 
 * @return false 
 */
bool gaussianFilter(){
    std::string srcPaht="D:/C++/gpu_conv_lib_cmake/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv_lib_cmake/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv_lib_cmake/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_shared.png";

    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    //================CPU进行高斯计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    gaussianConvolution(imgData.data.data(),out.data.data(),imgData.width,imgData.height, 7, 5.0f);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行高斯计算-全局内存=====================
    gaussianConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,7, 5.0f);
    out.imageSaveToFile(destPaht3);
    //================GPU进行高斯计算-共享内存=====================
    gaussianConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,7, 5.0f);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
    return true;
}
/**
 * @brief Sobel 边缘检测
 * @return true 
 * @return false 
 */
bool sobelFilter(){
    std::string srcPaht="D:/C++/gpu_conv_lib_cmake/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv_lib_cmake/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv_lib_cmake/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    //================CPU进行高斯计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    sobelConvolution(imgData.data.data(),out.data.data(),imgData.width,imgData.height, 1, 1);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行高斯计算-全局内存=====================
    sobelConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,1, 1);
    out.imageSaveToFile(destPaht3);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3},imgData.width,imgData.height);
    //================GPU进行高斯计算-共享内存=====================
    sobelConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,1, 1);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht4},imgData.width,imgData.height);
    return true;
}
/**
 * @brief Sobel 边缘检测（水平）（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool sobelXFilter(const std::string& src, const std::string& dest,Backend backend){
return true;
}
/**
 * @brief Sobel 边缘检测（垂直）（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool sobelYFilter(const std::string& src, const std::string& dest,Backend backend){
return true;
}
/**
 * @brief 锐化滤波器（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool sharpenFilter(){
    std::string srcPaht="D:/C++/gpu_conv_lib_cmake/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv_lib_cmake/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv_lib_cmake/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    //================CPU进行锐化计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    sharpenConvolution(imgData.data.data(),out.data.data(),imgData.width,imgData.height);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行锐化计算-全局内存=====================
    sharpenConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height);
    out.imageSaveToFile(destPaht3);
    //================GPU进行锐化计算-共享内存=====================
    sharpenConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
    return true;
}
/**
 * @brief 均值模糊滤波器（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool meanBlurFilter(){
    std::string srcPaht="D:/C++/gpu_conv_lib_cmake/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv_lib_cmake/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv_lib_cmake/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    //================CPU进行锐化计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    meanBlurConvolution(imgData.data.data(),out.data.data(),imgData.width,imgData.height,7);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行锐化计算-全局内存=====================
    meanBlurConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,7);
    out.imageSaveToFile(destPaht3);
    //================GPU进行锐化计算-共享内存=====================
    meanBlurConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,7);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
    return true;
}
/**
 * @brief 拉普拉斯算子（API）
 * @param src 原图图像路径
 * @param dest 保存图像路径
 * @param backend 
 * @return true 
 * @return false 
 */
bool laplacianFilter(){
    std::string srcPaht="D:/C++/gpu_conv_lib_cmake/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv_lib_cmake/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv_lib_cmake/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv_lib_cmake/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    //================CPU进行锐化计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    laplacianConvolution(imgData.data.data(),out.data.data(),imgData.width,imgData.height);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行锐化计算-全局内存=====================
    laplacianConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height);
    out.imageSaveToFile(destPaht3);
    //================GPU进行锐化计算-共享内存=====================
    laplacianConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
    return true;
}


} // gconv
