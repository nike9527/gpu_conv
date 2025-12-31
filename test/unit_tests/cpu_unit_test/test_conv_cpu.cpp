#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include "convolution_cpu.hpp"
#include "kernel.hpp"

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
        kernel = getKernel(params.kernel_type);
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
    std::vector<T> getKernel(int size, KernelType pattern = KernelType::FILTERKERNEL) {
        Kernel kernel;
        switch (pattern)
        {
        case KernelType::GAUSSIAN:
            kernel = Kernel::gaussian(size,0.5f);
            break;
        case KernelType::SOBELX:
             kernel = Kernel::sobelX();
            break;
        case KernelType::SOBELY:
             kernel = Kernel::sobelY();
            break;
        case KernelType::SHARPEN:
             kernel = Kernel::sharpen();
            break;
        case KernelType::MEANBLUR:
             kernel = Kernel::meanBlur(size);
            break;
        case KernelType::LAPLACIAN:
             kernel = Kernel::laplacian();
            break;
        case KernelType::FILTERKERNEL:
             kernel = Kernel::filterKernel(size ,std::vector<float>(size*size, 1.0f / (size * size)));
            break;
        default:
            kernel = Kernel::filterKernel(size , std::vector<float>(size*size, 1.0f / (size * size)) );
            break;
        }
        return kernel.kdata;
    }
    int num_elements;
    PerformanceTestParams params;
    std::vector<T> input;
    std::vector<T> output;
    std::vector<T> kernel;
};
/**
 * @brief Construct a new instantiate test suite p object
 * PerformanceTests 测试套件名称
 * Conv2dPerformanceTest 测试类
 * Values 参数列表
 */
using cpuConv2dPerformanceTestFloat = cpuConv2dPerformanceTest<float>;
INSTANTIATE_TEST_SUITE_P(PerformanceTests,cpuConv2dPerformanceTestFloat,testing::Values(
    PerformanceTestParams{512, 512, 3, 10, KernelType::GAUSSIAN}
    // PerformanceTestParams{256, 512, 5, 10, KernelType::SOBELX},
    // PerformanceTestParams{512, 512, 5, 10, KernelType::SOBELY},
    // PerformanceTestParams{512, 512, 5, 10, KernelType::MEANBLUR},
    // PerformanceTestParams{512, 512, 5, 10, KernelType::LAPLACIAN}
));

TEST_P(cpuConv2dPerformanceTestFloat, PerformanceBenchmark) {
     // 预热
    conv2dCpuOmp(input.data(), output.data(), params.width, params.height, params.ksize, kernel.data());
    
    auto total_duration = std::chrono::milliseconds(0);
    
    for (int i = 0; i < params.iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        conv2dCpuOmp(input.data(), output.data(), params.width, params.height, 
                    params.ksize, kernel.data());
        auto end = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    
    // 计算平均时间
    double avg_time_ms = static_cast<double>(total_duration.count()) / params.iterations;
    
    // 计算吞吐量（百万像素/秒）
    double mega_pixels = (params.width * params.height) / 1e6;
    double throughput = mega_pixels / (avg_time_ms / 1000.0);
    
    // 输出性能信息
    std::cout << "Performance - " << params.width << "x" << params.height 
              << ", Kernel: " << static_cast<int>(params.kernel_type)
              << ", Size: " << params.ksize
              << " - Avg: " << avg_time_ms << " ms, " 
              << throughput << " MP/s" << std::endl;
    
    // 可以设置性能阈值
    EXPECT_LT(avg_time_ms, 1000.0) << "Convolution too slow!";
}

