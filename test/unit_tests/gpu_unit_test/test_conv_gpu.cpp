#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include "convolution_gpu.hpp"
#include "kernel.hpp"
#include <iomanip> 

/**
 * @brief 参数
 * width 图像宽度
 * height 图像高度
 * ksize 内核大小
 * kernel 内核
 * iterations 迭代次数
 */
struct PerformanceTestParams {int width;int height; int iterations; int ksize; KernelType kernel_type; int block_w; int block_h;};
template<typename T>
class gpuConv2dPerformanceTest : public testing::TestWithParam<PerformanceTestParams> {
protected:
    void SetUp() override {
        params = GetParam();
        num_elements = params.width * params.height;
        // 创建测试数据
        input = createTestImage(num_elements);
        output.resize(num_elements, 0.0f);
        kernel = getKernel(params.ksize,params.kernel_type);
    }
     // 辅助函数：创建测试图像
    std::vector<T> createTestImage(int size, int pattern = 0) {
        std::vector<T> img(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-1.0f, 1.0f);
        for (int i = 0; i < num_elements; ++i) {
            img[i] = dis(gen);
        }
        return img;
    }
    // 辅助函数：创建kernel
    std::vector<T> getKernel(int size, KernelType pattern) {
        Kernel kernel;
        switch (pattern){
            case KernelType::GAUSSIAN: kernel = Kernel::gaussian(size,0.5f);break;
            case KernelType::SOBELX:kernel = Kernel::sobelX();break;
            case KernelType::SOBELY:kernel = Kernel::sobelY();break;
            case KernelType::SHARPEN:kernel = Kernel::sharpen();break;
            case KernelType::MEANBLUR:kernel = Kernel::meanBlur(size);break;
            case KernelType::LAPLACIAN:kernel = Kernel::laplacian();break;
            case KernelType::FILTERKERNEL:kernel = Kernel::filterKernel(size ,std::vector<float>(size*size, 1.0f / (size * size)));break;
            default:kernel = Kernel::filterKernel(size , std::vector<float>(size*size, 1.0f / (size * size)) );break;
        }
        return kernel.kdata;
    }
    // 计算合理的性能阈值
    double calculatePerformanceThreshold() const{
        // ===============================
        // 1. 基准吞吐（MP/s）——来自真实 GPU 实测
        //    仅使用 >= 1MP 的 case
        // ===============================
        static const std::map<std::tuple<int,int,int>, double> baseline_table = {
            // Gaussian (Global Memory)
            {{1024, 1024, 3}, 800.0},
            {{1024, 1024, 5}, 740.0},
        };

        const auto key = std::make_tuple(params.width, params.height, params.ksize);
        const double mega_pixels =
            static_cast<double>(params.width) * params.height / 1e6;

        double base_throughput = 0.0;

        // ===============================
        // 2. 查表（优先）
        // ===============================
        auto it = baseline_table.find(key);
        if (it != baseline_table.end())
        {
            base_throughput = it->second;
        }
        else
        {
            // ===============================
            // 3. 回退模型（只允许向下保守估计）
            // ===============================
            // 以 3x3 作为 reference
            constexpr double ref_3x3 = 800.0;

            base_throughput =
                ref_3x3 * (9.0 / (params.ksize * params.ksize));

            // 小图像：计时不准，主动放宽阈值
            if (mega_pixels < 1.0)
            {
                base_throughput *= 0.5;  // 放宽 50%
            }
            else if (mega_pixels > 2.0)
            {
                // 大图像：更好地利用带宽
                base_throughput *= 1.05;
            }
        }

        // ===============================
        // 4. Kernel 类型修正
        // ===============================
        double kernel_factor = 1.0;
        switch (params.kernel_type)
        {
            case KernelType::SOBELX:
            case KernelType::SOBELY:
                kernel_factor = 1.15;  // 稀疏 / 更少乘法
                break;

            case KernelType::SHARPEN:
            case KernelType::LAPLACIAN:
                kernel_factor = 1.05;
                break;

            case KernelType::GAUSSIAN:
                kernel_factor = 1.0;
                break;

            case KernelType::MEANBLUR:
                kernel_factor = 0.95;
                break;

            default:
                kernel_factor = 1.0;
        }

        // ===============================
        // 5. CI 安全边际（关键）
        //    只要掉 40% 就 fail
        // ===============================
        constexpr double ci_margin = 0.6;

        return base_throughput * kernel_factor * ci_margin;
    }

   //生成csv报告
   void genCsvReport(float avg_time_ms, float throughput) const {
        // CSV 格式输出
        std::cout << "PERF_CSV," 
              << params.width << ","
              << params.height << ","
              << Kernel::getKernelName(params.kernel_type)<< ","
              << params.ksize << ","
              << params.iterations << ","
              << std::fixed << std::setprecision(3) << avg_time_ms << " ms,"
              << throughput << std::endl;
                // 同时写入文件
            std::ofstream csv_file("performance_results.csv", std::ios::app);
            if (csv_file.is_open()) {
                csv_file << params.width << ","
                        << params.height << ","
                        << static_cast<int>(params.kernel_type) << ","
                        << params.ksize << ","
                        << params.iterations << ","
                        << avg_time_ms << ","
                        << throughput << "\n";
    }
   }
    int num_elements;
    PerformanceTestParams params;
    std::vector<T> input;
    std::vector<T> output;
    std::vector<T> kernel;
};
/**
 * @brief Construct a new instantiate test suite p object
 * gpuConv2dPerformanceTestFloat 测试套件名称
 * Conv2dPerformanceTest 测试类
 * Values 参数列表
 */
using gpuConv2dPerformanceTestFloat = gpuConv2dPerformanceTest<float>;
/**
 * @brief 高斯测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsGAUSSIAN,gpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::GAUSSIAN,16,16},
    PerformanceTestParams{512, 512, 10, 5, KernelType::GAUSSIAN,16,16},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::GAUSSIAN,16,16},
    PerformanceTestParams{1024, 1024, 10, 5, KernelType::GAUSSIAN,16,16}
));
/**
 * @brief 锐化滤波器测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsSHARPEN,gpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::SHARPEN,16,16},
    PerformanceTestParams{512, 512, 10, 3, KernelType::SHARPEN,16,16},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::SHARPEN,16,16},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::SHARPEN,16,16}
));
/**
 * @brief 均值模糊滤波器测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsMEANBLUR,gpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::MEANBLUR,16,16},
    PerformanceTestParams{512, 512, 10, 5, KernelType::MEANBLUR,16,16},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::MEANBLUR,16,16},
    PerformanceTestParams{1024, 1024, 10, 5, KernelType::MEANBLUR,16,16}
));
/**
 * @brief 拉普拉斯测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsLAPLACIAN,gpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::LAPLACIAN,16,16},
    PerformanceTestParams{512, 512, 10, 3, KernelType::LAPLACIAN,16,16},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::LAPLACIAN,16,16},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::LAPLACIAN,16,16}
));

/**
 * @brief 自定义内核测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsFILTERKERNEL,gpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::FILTERKERNEL,16,16},
    PerformanceTestParams{512, 512, 10, 5, KernelType::FILTERKERNEL,16,16},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::FILTERKERNEL,16,16},
    PerformanceTestParams{1024, 1024, 10, 5, KernelType::FILTERKERNEL,16,16}
));

TEST_P(gpuConv2dPerformanceTestFloat, PerformanceGlobalMem) {
     // 预热
    conv2dGlobalGPU(input.data(), output.data(), params.width, params.height, params.ksize, kernel.data(),params.block_w,params.block_h);
    
    auto total_duration = std::chrono::milliseconds(0);
    
    for (int i = 0; i < params.iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        conv2dGlobalGPU(input.data(), output.data(), params.width, params.height, params.ksize, kernel.data(),params.block_w,params.block_h);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    
    // 计算平均时间
    double avg_time_ms = static_cast<double>(total_duration.count()) / params.iterations;
    
    // 计算吞吐量（百万像素/秒）
    double mega_pixels = (params.width * params.height) / 1e6;
    double throughput = mega_pixels / (avg_time_ms / 1000.0);
    // 计算动态阈值
    double threshold_mp_s  = calculatePerformanceThreshold();
    // 输出详细性能报告
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Performance Report (GPU-Global-memory)  " << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "block size:      " << params.block_w << "×" << params.block_h << std::endl;
    std::cout << "Resolution:      " << params.width << " × " << params.height 
              << " (" << (params.width * params.height / 1e6) << " MP)" << std::endl;
    std::cout << "Kernel:          " << Kernel::getKernelName(params.kernel_type) 
              << " " << params.ksize << "×" << params.ksize << std::endl;
    std::cout << "Computer:        " << std::fixed << std::setprecision(1)
              << (params.width * params.height * params.ksize * params.ksize / 1e6) 
              << " million times" << std::endl;
    std::cout << "Avg time:        " << std::fixed << std::setprecision(1)
              << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput:      " << std::fixed << std::setprecision(1)
              << throughput << " MP/s" << std::endl;
    std::cout << "Threshold:       " << std::fixed << std::setprecision(1)
              << threshold_mp_s << " ms" << std::endl;
    std::cout << "Result:          " 
              << (avg_time_ms < threshold_mp_s ? "PASS" : "FAIL") 
              << std::endl;
    std::cout << "========================================\n" << std::endl;
    //设置性能阈值(性能下限)
    // EXPECT_LT(avg_time_ms, 40.0f) << "Convolution too slow!";
    // 输出性能建议 
    EXPECT_GT(throughput, threshold_mp_s) << "性能警告: " << avg_time_ms << " ms > 建议阈值 "<< threshold_mp_s << " MP/s";
}

TEST_P(gpuConv2dPerformanceTestFloat, PerformanceSharedConst) {
     // 预热
    conv2dWithSharedGPU(input.data(), output.data(), params.width, params.height, params.ksize, kernel.data(),params.block_w,params.block_h);
    
    auto total_duration = std::chrono::milliseconds(0);
    
    for (int i = 0; i < params.iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        conv2dWithSharedGPU(input.data(), output.data(), params.width, params.height, params.ksize, kernel.data(),params.block_w,params.block_h);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    
    // 计算平均时间
    double avg_time_ms = static_cast<double>(total_duration.count()) / params.iterations;
    
    // 计算吞吐量（百万像素/秒）
    double mega_pixels = (params.width * params.height) / 1e6;
    double throughput = mega_pixels / (avg_time_ms / 1000.0);
    // 计算动态阈值
    double threshold_mp_s  = calculatePerformanceThreshold();
    // 输出详细性能报告
    std::cout << "\n========================================" << std::endl;
    std::cout << " Performance Report (GPU-Shared-Constant) " << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "block size:      " << params.block_w << "×" << params.block_h << std::endl;
    std::cout << "Resolution:      " << params.width << " × " << params.height 
              << " (" << (params.width * params.height / 1e6) << " MP)" << std::endl;
    std::cout << "Kernel:          " << Kernel::getKernelName(params.kernel_type) 
              << " " << params.ksize << "×" << params.ksize << std::endl;
    std::cout << "Computer:        " << std::fixed << std::setprecision(1)
              << (params.width * params.height * params.ksize * params.ksize / 1e6) 
              << " million times" << std::endl;
    std::cout << "Avg time:        " << std::fixed << std::setprecision(1)
              << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput:      " << std::fixed << std::setprecision(1)
              << throughput << " MP/s" << std::endl;
    std::cout << "Threshold:       " << std::fixed << std::setprecision(1)
              << threshold_mp_s << " ms" << std::endl;
    std::cout << "Result:          " 
              << (avg_time_ms < threshold_mp_s ? "PASS" : "FAIL") 
              << std::endl;
    std::cout << "========================================\n" << std::endl;
    //设置性能阈值(性能下限)
    // EXPECT_LT(avg_time_ms, 40.0f) << "Convolution too slow!";
    // 输出性能建议 
    EXPECT_GT(throughput, threshold_mp_s) << "性能警告: " << avg_time_ms << " ms > 建议阈值 "<< threshold_mp_s << " MP/s";
}

