
#include <iostream>
#include <chrono>
#include "action.hpp"
#include "image_codec.hpp"
#include "image_viewer.hpp"
#include "filters/filter.hpp"
#include "conv/conv_cpu.hpp"
#include "conv/conv_gpu.hpp"
#include <cuda_runtime.h>

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
        // filter filter = filter::gaussian2D(5, 5.0f);
        // filter filter = filter::sharpen();
        // filter filter = filter::meanBlur(9);
        filter filter = filter::laplacian();
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
        //================CPU进行高斯计算=====================
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_conv::gaussianBlur2D(imgData.data.data(), out.data.data(), imgData.width, imgData.height, 7, 5.f);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
        out.imageSaveToFile(destPaht2);
        //================GPU进行高斯计算-全局内存=====================
        gpu_conv::gaussianBlur(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::GLOBAL, 7, 5.f);
        out.imageSaveToFile(destPaht3);
        //================GPU进行高斯计算-共享内存=====================
        gpu_conv::gaussianBlur(imgData.data.data(), out.data.data(), imgData.width, imgData.height, mem_type::SHAREDCONST, 7, 5.f);
        out.imageSaveToFile(destPaht4);

        gpu_conv::launchGaussianRGBA(out.data.data());

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
    // bool pipelineAction()
    // {
    //     // std::string srcPaht = "D:/C++/gpu_conv/image/lena.png";
    //     // std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGary.png";
    //     // std::string destPaht2 = "D:/C++/gpu_conv/image/lenaCPU.png";
    //     // std::string destPaht3 = "D:/C++/gpu_conv/image/lenaGPU_global.png";
    //     // std::string destPaht4 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
    //     // Image imgData = Image::imageLoadGray(srcPaht);
    //     // Image out(imgData.width, imgData.height);
    //     // imgData.imageSaveToGray(destPaht1);
    //     // //================GPU进行计算-共享内存=====================
    //     // filter_pipeline pipe(imgData.width, imgData.height, cudaStreamNonBlocking);
    //     // gpu_conv::launchFilterAsync(pipe, imgData.data.data(), out.data.data(), imgData.width, imgData.height, filter_type::GAUSSIAN);
    //     // out.imageSaveToFile(destPaht4);
    //     // renderImage(std::vector<std::string>{destPaht1, destPaht2, destPaht3, destPaht4}, imgData.width, imgData.height);
    //     // pipe.stream.synchronize();

    //     std::string srcPaht1 = "D:/C++/gpu_conv/image/lena.png";
    //     std::string srcPaht2 = "D:/C++/gpu_conv/image/16.png";
    //     std::string srcPaht3 = "D:/C++/gpu_conv/image/17.png";
    //     std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
    //     std::string destPaht2 = "D:/C++/gpu_conv/image/16GPU_shared.png";
    //     std::string destPaht3 = "D:/C++/gpu_conv/image/17GPU_shared.png";
    //     std::vector<std::string> path{
    //         destPaht1,
    //         destPaht2,
    //         destPaht3};
    //     std::vector<std::string> src{
    //         srcPaht1,
    //         srcPaht2,
    //         srcPaht3};
    //     //================GPU进行计算-共享内存=====================
    //     filter_pipeline pipe(512, 512, cudaStreamNonBlocking);
    //     for (int frame = 0; frame < 3; ++frame)
    //     {
    //         Image imgData = Image::imageLoadGray(src[frame]);
    //         Image out(512, 512);
    //         gpu_conv::launchFilterAsync(pipe, imgData.data.data(), out.data.data(), pipe.width, pipe.height, filter_type::GAUSSIAN);
    //         out.imageSaveToFile(path[frame]);
    //     }
    //     renderImage(path, 800, 800);
    //     pipe.stream.synchronize();
    //     return true;
    // }

    // bool pipelineAction()
    // {
    //     std::string srcPaht1 = "D:/C++/gpu_conv/image/lena.png";
    //     std::string srcPaht2 = "D:/C++/gpu_conv/image/16.png";
    //     std::string srcPaht3 = "D:/C++/gpu_conv/image/17.png";
    //     std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
    //     std::string destPaht2 = "D:/C++/gpu_conv/image/16GPU_shared.png";
    //     std::string destPaht3 = "D:/C++/gpu_conv/image/17GPU_shared.png";

    //     std::vector<std::string> src_paths = {srcPaht1, srcPaht2, srcPaht3};
    //     std::vector<std::string> dest_paths = {destPaht1, destPaht2, destPaht3};

    //     filter_pipeline pipe(800, 800, cudaStreamDefault);

    //     for (size_t frame = 0; frame < src_paths.size(); ++frame)
    //     {
    //         // 1. 加载输入图像
    //         Image input_img = Image::imageLoadGray(src_paths[frame]);

    //         // 2. 准备输出图像（同样大小）
    //         Image output_img(input_img.width, input_img.height);

    //         // 3. 启动GPU滤波（同步版本，或异步+立即同步）
    //         gpu_conv::launchFilterAsync(pipe,
    //                                     input_img.data.data(),
    //                                     output_img.data.data(),
    //                                     input_img.width,
    //                                     input_img.height,
    //                                     filter_type::GAUSSIAN);

    //         // 4. 保存处理后的图像
    //         output_img.imageSaveToFile(dest_paths[frame]);

    //         std::cout << "Processed frame " << frame << ": " << src_paths[frame] << " -> " << dest_paths[frame] << std::endl;
    //     }

    //     // 渲染所有处理后的图像
    //     renderImage(dest_paths, 800, 800);

    //     return true;
    // }

    bool pipelineAction()
    {
        /* std::string srcPaht1 = "D:/C++/gpu_conv/image/lena.png";
         std::string srcPaht2 = "D:/C++/gpu_conv/image/16.png";
         std::string srcPaht3 = "D:/C++/gpu_conv/image/17.png";
         std::string destPaht1 = "D:/C++/gpu_conv/image/lenaGPU_shared.png";
         std::string destPaht2 = "D:/C++/gpu_conv/image/16GPU_shared.png";
         std::string destPaht3 = "D:/C++/gpu_conv/image/17GPU_shared.png";

         std::vector<std::string> src_paths = {srcPaht1, srcPaht2, srcPaht3};
         std::vector<std::string> dest_paths = {destPaht1, destPaht2, destPaht3};

         // 1. 预加载所有输入图像（CPU，可以并行）
         std::vector<Image> input_images;
         for (const auto &path : src_paths)
         {
             input_images.push_back(Image::imageLoadGray(path));
         }

         // 2. 准备输出图像容器
         std::vector<Image> output_images;
         for (const auto &img : input_images)
         {
             output_images.emplace_back(img.width, img.height);
         }

         // 3. 创建多个流用于真正的异步流水线
         const int num_streams = 2; // 根据GPU能力调整
         std::vector<filter_pipeline> pipes;

         for (int i = 0; i < num_streams; ++i)
         {
             cudaStream_t stream;
             pipes.emplace_back(800, 800, cudaStreamNonBlocking);
         }

         // 4. 异步处理所有帧
         std::vector<cudaEvent_t> frame_events(src_paths.size());
         for (size_t i = 0; i < frame_events.size(); ++i)
         {
             cudaEventCreate(&frame_events[i]);
         }

         for (size_t frame = 0; frame < src_paths.size(); ++frame)
         {
             int stream_idx = frame % num_streams;

             // 异步启动GPU处理
             gpu_conv::launchFilterAsync(pipes[stream_idx],
                                         input_images[frame].data.data(),
                                         output_images[frame].data.data(),
                                         input_images[frame].width,
                                         input_images[frame].height,
                                         filter_type::GAUSSIAN);

             // 记录事件，标记该帧处理完成
             cudaEventRecord(frame_events[frame], pipes[stream_idx].stream.get());
         }

         // 5. 等待所有帧处理完成
         for (size_t frame = 0; frame < src_paths.size(); ++frame)
         {
             cudaEventSynchronize(frame_events[frame]);

             // 现在可以安全保存
             output_images[frame].imageSaveToFile(dest_paths[frame]);
             std::cout << "Saved frame " << frame << ": " << dest_paths[frame] << std::endl;

             cudaEventDestroy(frame_events[frame]);
         }

         cudaDeviceSynchronize();

         // 7. 渲染结果
         renderImage(dest_paths, 800, 800);
*/
        return true;
    }
    /**
     * @brief
 FilterPipeline pipe[2] = {
    {width, height},
    {width, height}
};

for (int frame = 0; frame < N; ++frame) {
    int i = frame & 1;

    launchFilterAsync(
        pipe[i],
        input[frame],
        output[frame],
        mem_type::SHAREDCONST,
        filterObj,
        16, 16
    );
}

// 同步最后一帧
pipe[(N - 1) & 1].stream.synchronize();
     *
     */
} // gconv
