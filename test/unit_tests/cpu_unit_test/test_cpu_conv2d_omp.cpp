#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include "convolution_cpu.hpp"
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
struct PerformanceTestParams {int width;int height; int iterations; int ksize; KernelType kernel_type;};
template<typename T>
class cpuConv2dPerformanceTest : public testing::TestWithParam<PerformanceTestParams> {
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
    double calculatePerformanceThreshold() const {
       // 基于实际测试数据的查找表 {width, height, ksize} -> 吞吐量 (MP/s)
        static const std::map<std::tuple<int, int, int>, double> baseline_table = {
            {{512, 512, 3}, 45.0},{{512, 512, 5}, 16.0},
            {{1024, 1024, 3}, 45.0},{{1024, 1024, 5}, 19.0},
            {{256, 256, 3}, 45.0},{{256, 256, 5}, 16.0},
            {{128, 128, 3}, 45.0},{{128, 128, 5}, 16.0},
        };
        
        // 尝试从表中查找
        auto key = std::make_tuple(params.width, params.height, params.ksize);
        auto it = baseline_table.find(key);
        
        double base_throughput;
        if (it != baseline_table.end()) {
            base_throughput = it->second;
        } else {
            //如果没有匹配，使用经验公式
            double mega_pixels = (params.width * params.height) / 1e6;
            
            if (params.ksize == 3) {
                base_throughput = 45.0;  // 3x3 基准
            } else if (params.ksize == 5) {
                base_throughput = 45.0 * (9.0 / 25.0);  // 按面积缩放
            } else {
                base_throughput = 45.0 * (9.0 / (params.ksize * params.ksize));
            }
            // 大图像可能有更好的缓存利用率
            if (mega_pixels > 1.0) {
                base_throughput *= 1.1;  // 大图像 +10%
            }
        }
        
        // 内核类型调整
        double kernel_factor = 1.0;
        switch (params.kernel_type) {
            case KernelType::SOBELX:
            case KernelType::SOBELY:
                kernel_factor = 1.2;   // 稀疏，更快
                break;
            case KernelType::SHARPEN:
            case KernelType::LAPLACIAN:
                kernel_factor = 1.05;  // 稍快
                break;
            case KernelType::GAUSSIAN:
                kernel_factor = 1.0;   // 基准
                break;
            case KernelType::MEANBLUR:
                kernel_factor = 0.95;  // 稍慢
                break;
            default:
                kernel_factor = 1.0;
        }
        
        // 4. CI 环境安全边际
        double ci_margin = 0.85;
        
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
 * cpuConv2dPerformanceTestFloat 测试套件名称
 * Conv2dPerformanceTest 测试类
 * Values 参数列表
 */
using cpuConv2dPerformanceTestFloat = cpuConv2dPerformanceTest<float>;
/**
 * @brief 高斯测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsGAUSSIAN,cpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::GAUSSIAN},
    PerformanceTestParams{512, 512, 10, 5, KernelType::GAUSSIAN},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::GAUSSIAN},
    PerformanceTestParams{1024, 1024, 10, 5, KernelType::GAUSSIAN}
));
/**
 * @brief 锐化滤波器测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsSHARPEN,cpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::SHARPEN},
    PerformanceTestParams{512, 512, 10, 3, KernelType::SHARPEN},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::SHARPEN},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::SHARPEN}
));
/**
 * @brief 均值模糊滤波器测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsMEANBLUR,cpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::MEANBLUR},
    PerformanceTestParams{512, 512, 10, 5, KernelType::MEANBLUR},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::MEANBLUR},
    PerformanceTestParams{1024, 1024, 10, 5, KernelType::MEANBLUR}
));
/**
 * @brief 拉普拉斯测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsLAPLACIAN,cpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::LAPLACIAN},
    PerformanceTestParams{512, 512, 10, 3, KernelType::LAPLACIAN},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::LAPLACIAN},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::LAPLACIAN}
));

/**
 * @brief 自定义内核测试套件
 */
INSTANTIATE_TEST_SUITE_P(PerformanceTestsFILTERKERNEL,cpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 10, 3, KernelType::FILTERKERNEL},
    PerformanceTestParams{512, 512, 10, 5, KernelType::FILTERKERNEL},
    PerformanceTestParams{1024, 1024, 10, 3, KernelType::FILTERKERNEL},
    PerformanceTestParams{1024, 1024, 10, 5, KernelType::FILTERKERNEL}
));

TEST_P(cpuConv2dPerformanceTestFloat, PerformanceConv2d) {
     // 预热
    conv2dCpuOmp(input.data(), output.data(), params.width, params.height, params.ksize, kernel.data());
    
    auto total_duration = std::chrono::milliseconds(0);
    
    for (int i = 0; i < params.iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        conv2dCpuOmp(input.data(), output.data(), params.width, params.height, params.ksize, kernel.data());
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
    std::cout << "         Performance Report (CPU)"          << std::endl;
    std::cout << "==========================================" << std::endl;
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

