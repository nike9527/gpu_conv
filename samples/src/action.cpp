
#include <iostream>
#include <chrono>
#include "action.hpp"
#include "image_codec.hpp"
#include "image_viewer.hpp"
#include "filters/filter.hpp"
#include "conv/conv_cpu.hpp"
#include "conv/conv_gpu.hpp"

namespace gconv
{

    /**
     * @brief 自定义卷积函数
     * @param src 原图图像路径
     * @param dest 保存图像路径
     * @param k 内核矩阵
     * @param backend
     * @return true
     * @return false
     */
    bool convolve()
    {
        std::string srcPaht = "D:/C++/gpu_conv/image/lena.png";
        std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGary.png";
        std::string destPaht2 = "D:/C++/gpu_conv/image/lenaCPU.png";
        std::string destPaht3 = "D:/C++/gpu_conv/image/lenaGPU_global.png";
        std::string destPaht4 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
        Image imgData = Image::imageLoadGray(srcPaht);
        Image out(imgData.width, imgData.height);
        imgData.imageSaveToGray(destPaht1);
        filter filter = filter::gaussian2D(5, 5.0f);
        // filter filter = filter::sharpen();
        // filter filter = filter::meanBlur(9);
        // filter filter = filter::laplacian();
        //================CPU进行卷积计算=====================
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_conv::conv2dKernel(imgData.data.data(), out.data.data(), imgData.width, imgData.height, filter);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        out.imageSaveToFile(destPaht2);
        //================GPU进行卷积计算-全局内存=====================
        gpu_conv::conv2dKernel(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::GLOBAL, filter);
        out.imageSaveToFile(destPaht3);
        //================GPU进行卷积计算-共享内存=====================
        gpu_conv::conv2dKernel(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::SHAREDCONST, filter);
        out.imageSaveToFile(destPaht4);
        renderImage(std::vector<std::string>{destPaht1, destPaht2, destPaht3, destPaht4}, imgData.width, imgData.height);
        return true;
    }
    /**
     * @brief 高斯滤波入口（API）
     * @return true
     * @return false
     */
    bool gaussianAction()
    {
        std::string srcPaht = "D:/C++/gpu_conv/image/lena.png";
        std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGary.png";
        std::string destPaht2 = "D:/C++/gpu_conv/image/lenaCPU.png";
        std::string destPaht3 = "D:/C++/gpu_conv/image/lenaGPU_global.png";
        std::string destPaht4 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";

        Image imgData = Image::imageLoadGray(srcPaht);
        Image out(imgData.width, imgData.height);
        imgData.imageSaveToGray(destPaht1);
        filter filter = filter::gaussian2D(3, .5f);
        //================CPU进行高斯计算=====================
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_conv::gaussianBlur2D(imgData.data.data(), out.data.data(), imgData.width, imgData.height, 3, 5.f);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        out.imageSaveToFile(destPaht2);
        //================GPU进行高斯计算-全局内存=====================
        gpu_conv::gaussianBlur(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::GLOBAL, 3, 5.f);
        out.imageSaveToFile(destPaht3);
        //================GPU进行高斯计算-共享内存=====================
        gpu_conv::gaussianBlur(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::SHAREDCONST, 3, 5.f);
        out.imageSaveToFile(destPaht4);
        renderImage(std::vector<std::string>{destPaht1, destPaht2, destPaht3, destPaht4}, imgData.width, imgData.height);
        return true;
    }
    /**
     *
     * @brief Sobel 边缘检测
     * @return true
     * @return false
     */
    bool sobelAction()
    {
        std::string srcPaht = "D:/C++/gpu_conv/image/lena.png";
        std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGary.png";
        std::string destPaht2 = "D:/C++/gpu_conv/image/lenaCPU.png";
        std::string destPaht3 = "D:/C++/gpu_conv/image/lenaGPU_global.png";
        std::string destPaht4 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
        Image imgData = Image::imageLoadGray(srcPaht);
        Image out(imgData.width, imgData.height);
        imgData.imageSaveToGray(destPaht1);
        filter sobel = filter::sobelX();
        //================CPU进行高斯计算=====================
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_conv::sobel(imgData.data.data(), out.data.data(), imgData.width, imgData.height);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        out.imageSaveToFile(destPaht2);
        //================GPU进行高斯计算-全局内存=====================
        gpu_conv::sobel(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::GLOBAL);
        out.imageSaveToFile(destPaht3);
        //================GPU进行高斯计算-共享内存=====================
        gpu_conv::sobel(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::SHAREDCONST);
        out.imageSaveToFile(destPaht4);
        renderImage(std::vector<std::string>{destPaht1, destPaht2, destPaht3, destPaht4}, imgData.width, imgData.height);
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
    bool sobelXAction(const std::string &src, const std::string &dest, Backend backend)
    {
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
    bool sobelYAction(const std::string &src, const std::string &dest, Backend backend)
    {
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
    bool sharpenAction()
    {
        std::string srcPaht = "D:/C++/gpu_conv/image/lena.png";
        std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGary.png";
        std::string destPaht2 = "D:/C++/gpu_conv/image/lenaCPU.png";
        std::string destPaht3 = "D:/C++/gpu_conv/image/lenaGPU_global.png";
        std::string destPaht4 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
        Image imgData = Image::imageLoadGray(srcPaht);
        Image out(imgData.width, imgData.height);
        imgData.imageSaveToGray(destPaht1);
        filter sharpen = filter::sharpen();
        //================CPU进行锐化计算=====================
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_conv::sharpen(imgData.data.data(), out.data.data(), imgData.width, imgData.height);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        out.imageSaveToFile(destPaht2);
        //================GPU进行锐化计算-全局内存=====================
        gpu_conv::sharpen(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::GLOBAL);
        out.imageSaveToFile(destPaht3);
        //================GPU进行锐化计算-共享内存=====================
        gpu_conv::sharpen(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::SHAREDCONST);
        out.imageSaveToFile(destPaht4);
        renderImage(std::vector<std::string>{destPaht1, destPaht2, destPaht3, destPaht4}, imgData.width, imgData.height);
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
    bool meanBlurAction()
    {
        std::string srcPaht = "D:/C++/gpu_conv/image/lena.png";
        std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGary.png";
        std::string destPaht2 = "D:/C++/gpu_conv/image/lenaCPU.png";
        std::string destPaht3 = "D:/C++/gpu_conv/image/lenaGPU_global.png";
        std::string destPaht4 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
        Image imgData = Image::imageLoadGray(srcPaht);
        Image out(imgData.width, imgData.height);
        imgData.imageSaveToGray(destPaht1);
        filter meanBlur = filter::meanBlur(3);
        //================CPU进行锐化计算=====================
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_conv::meanBlur(imgData.data.data(), out.data.data(), imgData.width, imgData.height, 5);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        out.imageSaveToFile(destPaht2);
        //================GPU进行锐化计算-全局内存=====================
        gpu_conv::meanBlur(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::GLOBAL, 5);
        out.imageSaveToFile(destPaht3);
        //================GPU进行锐化计算-共享内存=====================
        gpu_conv::meanBlur(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::SHAREDCONST, 5);
        out.imageSaveToFile(destPaht4);
        renderImage(std::vector<std::string>{destPaht1, destPaht2, destPaht3, destPaht4}, imgData.width, imgData.height);
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
    bool laplacianAction()
    {
        std::string srcPaht = "D:/C++/gpu_conv/image/lena.png";
        std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGary.png";
        std::string destPaht2 = "D:/C++/gpu_conv/image/lenaCPU.png";
        std::string destPaht3 = "D:/C++/gpu_conv/image/lenaGPU_global.png";
        std::string destPaht4 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
        Image imgData = Image::imageLoadGray(srcPaht);
        Image out(imgData.width, imgData.height);
        imgData.imageSaveToGray(destPaht1);
        filter laplacian = filter::laplacian();
        //================CPU进行锐化计算=====================
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_conv::laplacian(imgData.data.data(), out.data.data(), imgData.width, imgData.height);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        out.imageSaveToFile(destPaht2);
        //================GPU进行锐化计算-全局内存=====================
        gpu_conv::laplacian(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::GLOBAL);
        out.imageSaveToFile(destPaht3);
        //================GPU进行锐化计算-共享内存=====================
        gpu_conv::laplacian(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::SHAREDCONST);
        out.imageSaveToFile(destPaht4);
        renderImage(std::vector<std::string>{destPaht1, destPaht2, destPaht3, destPaht4}, imgData.width, imgData.height);
        return true;
    }
    bool conv2dWithAsync()
    {
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
