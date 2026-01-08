
#include <iostream>
#include <chrono>
#include "action.hpp"
#include "image_codec.hpp"
#include "image_viewer.hpp"
#include "filters/filter.hpp"

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
    std::string srcPaht="D:/C++/gpu_conv/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    filter filter = filter::gaussian(5, 5.0f);
    // filter filter = filter::sharpen();
    // filter filter = filter::meanBlur(9);
    // filter filter = filter::laplacian();
    kernel_filter kernelFilter(filter.kdata.data(),filter.size);
    //================CPU进行卷积计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    kernelFilter.conv2dCpuOmp(imgData.data.data(),out.data.data(),imgData.width,imgData.height, filter);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行卷积计算-全局内存=====================
    kernelFilter.conv2dGlobalGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height, filter);
    out.imageSaveToFile(destPaht3);
    //================GPU进行卷积计算-共享内存=====================
    kernelFilter.conv2dGlobalGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height, filter);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
    return true;
}
/**
 * @brief 高斯滤波入口（API）
 * @return true 
 * @return false 
 */
bool gaussianAction(){
    std::string srcPaht="D:/C++/gpu_conv/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv/image/lenaGPU_shared.png";

    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    filter filter = filter::gaussian(3,.5f);
    kernel_gaussian kernelGaussian(filter.kdata.data(),filter.size);
    //================CPU进行高斯计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    kernelGaussian.gaussianConvolutionCPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height, filter);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行高斯计算-全局内存=====================
    kernelGaussian.gaussianConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,filter);
    out.imageSaveToFile(destPaht3);
    //================GPU进行高斯计算-共享内存=====================
    kernelGaussian.gaussianConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,filter);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
    return true;
}
/**
 * 
 * @brief Sobel 边缘检测
 * @return true 
 * @return false 
 */
bool sobelAction(){
     std::string srcPaht="D:/C++/gpu_conv/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    filter sobel = filter::sobelX();
    kernel_sobel kernelSobel(sobel.kdata.data(),sobel.size);
    //================CPU进行高斯计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    kernelSobel.sobelConvolutionCPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height, 1, 1);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行高斯计算-全局内存=====================
    kernelSobel.sobelConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,1, 1);
    out.imageSaveToFile(destPaht3);
    //================GPU进行高斯计算-共享内存=====================
    kernelSobel.sobelConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,1, 1);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
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
bool sobelXAction(const std::string& src, const std::string& dest,Backend backend){
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
bool sobelYAction(const std::string& src, const std::string& dest,Backend backend){
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
bool sharpenAction(){
    std::string srcPaht="D:/C++/gpu_conv/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    filter sharpen = filter::sharpen();
    kernel_sharpen kernelSharpen(sharpen.kdata.data(),sharpen.size);
    //================CPU进行锐化计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    kernelSharpen.sharpenConvolutionCPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,sharpen);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行锐化计算-全局内存=====================
    kernelSharpen.sharpenConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,sharpen);
    out.imageSaveToFile(destPaht3);
    //================GPU进行锐化计算-共享内存=====================
    kernelSharpen.sharpenConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,sharpen);
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
bool meanBlurAction(){
    std::string srcPaht="D:/C++/gpu_conv/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    filter meanBlur = filter::meanBlur(3);
    kernel_meanBlur kernelMeanBlur(meanBlur.kdata.data(),meanBlur.size);
    //================CPU进行锐化计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    kernelMeanBlur.meanBlurConvolutionCPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,meanBlur);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行锐化计算-全局内存=====================
    kernelMeanBlur.meanBlurConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,meanBlur);
    out.imageSaveToFile(destPaht3);
    //================GPU进行锐化计算-共享内存=====================
    kernelMeanBlur.meanBlurConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,meanBlur);
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
bool laplacianAction(){
    std::string srcPaht="D:/C++/gpu_conv/image/lena.png";
    std::string destPaht1="D:/C++/gpu_conv/image/lenaGary.png";
    std::string destPaht2="D:/C++/gpu_conv/image/lenaCPU.png";
    std::string destPaht3="D:/C++/gpu_conv/image/lenaGPU_global.png";
    std::string destPaht4="D:/C++/gpu_conv/image/lenaGPU_shared.png";
    Image imgData = Image::imageLoadGray(srcPaht);
    Image out(imgData.width,imgData.height);
    imgData.imageSaveToGray(destPaht1);
    filter laplacian = filter::laplacian();
    kernel_laplacian kernelLaplacian(laplacian.kdata.data(),laplacian.size);
    //================CPU进行锐化计算=====================
    auto t1 = std::chrono::high_resolution_clock::now();
    kernelLaplacian.laplacianConvolutionCPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,laplacian);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    out.imageSaveToFile(destPaht2);
    //================GPU进行锐化计算-全局内存=====================
    kernelLaplacian.laplacianConvolutionGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,laplacian);
    out.imageSaveToFile(destPaht3);
    //================GPU进行锐化计算-共享内存=====================
    kernelLaplacian.laplacianConvolutionWithSharedGPU(imgData.data.data(),out.data.data(),imgData.width,imgData.height,laplacian);
    out.imageSaveToFile(destPaht4);
    renderImage(std::vector<std::string>{destPaht1,destPaht2,destPaht3,destPaht4},imgData.width,imgData.height);
    return true;
}
bool conv2dWithAsync(){
    // std::vector<std::string> srcPaht{"D:/C++/gpu_conv/image/lena.png","D:/C++/gpu_conv/image/16.png","D:/C++/gpu_conv/image/17.png"};
    // // std::vector<std::string> srcPaht{"D:/C++/gpu_conv/image/lenaGary.png","D:/C++/gpu_conv/image/lenaGPU_global.png","D:/C++/gpu_conv/image/lenaGPU_shared.png"};
    // std::vector<std::string> ourPaht{"D:/C++/gpu_conv/image/lena_2.png","D:/C++/gpu_conv/image/16_2.png","D:/C++/gpu_conv/image/17_2.png"};
    // std::vector<Image> inImg;
    // std::vector<Image> outImg;
    // for(std::string& path : srcPaht){
    //     Image imgData = Image::imageLoadGray(path);
    //     inImg.emplace_back(imgData);
    //     outImg.emplace_back(Image{imgData.width, imgData.height});
    // }
    // filter filter = filter::laplacian();
    // // filter filter = filter::gaussian(7,5.0);
    // // filter filter = filter::sharpen();
    // // filter filter = filter::meanBlur(7);
    // // filter filter = filter::laplacian();
    // conv2dWithAsyncGPU(inImg,outImg,filter.size,filter.kdata.data());
    // for(int i = 0; i< ourPaht.size(); i++){
    //     outImg[i].imageSaveToGray(ourPaht[i]);
    // }
    // renderImage(ourPaht,800,600);
    return true;
}

} // gconv
