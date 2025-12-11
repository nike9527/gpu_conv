#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <queue>
#include <memory>
#include <functional>

// 内存类型枚举
enum class memoryType {
    DEVICE,     // 设备内存
    HOST,       // 主机内存
    UNIFIED,    // 统一内存
    SHARED,     // 共享内存
    CONSTANT    // 常量内存
};

// 内存块信息结构
struct memoryBlock {
    void* ptr;              // 内存指针
    size_t size;           // 内存大小
    memoryType type;       // 内存类型
    bool is_allocated;     // 是否已分配
    cudaStream_t stream;   // 关联的流
    size_t alignment;      // 对齐要求
    
    memoryBlock() : ptr(nullptr), size(0), type(memoryType::DEVICE), 
                   is_allocated(false), stream(0), alignment(0) {}
};

// 异步操作结构
struct asyncOperation {
    enum class opType {
        COPY_H2D,
        COPY_D2H,
        COPY_D2D,
        KERNEL_LAUNCH,
        MEMSET
    };
    
    opType type;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    cudaStream_t stream;
    size_t bytes_transferred;
    std::function<void()> callback;
};

// CUDA内存管理主类
class memoryManager {
private:
    // 单例模式
    static memoryManager* instance;
    static std::mutex instance_mutex;
    
    // 内存池
    std::unordered_map<void*, memoryBlock> memory_pool;
    std::mutex pool_mutex;
    
    // 异步操作队列
    std::queue<asyncOperation> async_queue;
    std::mutex queue_mutex;
    
    // 常量内存缓存
    struct ConstantMemoryCache {
        std::unordered_map<std::string, std::pair<void*, size_t>> cache;
        std::mutex mutex;
        static constexpr size_t MAX_CONSTANT_SIZE = 65536; // 64KB
        size_t used_size = 0;
    } constant_cache;
    
    // 流管理
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    
    memoryManager();
    ~memoryManager();
    
public:
    // 获取单例
    static memoryManager* getInstance();
    
    // 禁止拷贝
    memoryManager(const memoryManager&) = delete;
    memoryManager& operator=(const memoryManager&) = delete;
    
    // 基础内存分配
    void* allocate(size_t size, memoryType type, size_t alignment = 256);
    void free(void* ptr);
    
    // 统一内存分配
    void* allocateUnified(size_t size, unsigned int flags = cudaMemAttachGlobal);
    
    // 共享内存分配（在kernel中使用，这里提供管理接口）
    template<typename T>
    T* getSharedMemory(size_t elements, cudaStream_t stream = 0);
    
    // 常量内存分配
    void* allocateConstant(size_t size, const std::string& key = "");
    
    // 异步数据拷贝
    void asyncCopy(void* dst, const void* src, size_t size, 
                   cudaMemcpyKind kind, cudaStream_t stream = 0,
                   std::function<void()> callback = nullptr);
    
    void asyncHostToDevice(void* dev_ptr, const void* host_ptr, 
                          size_t size, cudaStream_t stream = 0);
    void asyncDeviceToHost(void* host_ptr, const void* dev_ptr, 
                          size_t size, cudaStream_t stream = 0);
    
    // 流管理
    cudaStream_t createStream(bool non_blocking = false);
    void destroyStream(cudaStream_t stream);
    void synchronizeStream(cudaStream_t stream);
    void synchronizeAll();
    
    // 事件管理
    cudaEvent_t createEvent(bool timing = true, bool blocking = false);
    void recordEvent(cudaEvent_t event, cudaStream_t stream = 0);
    float getElapsedTime(cudaEvent_t start, cudaEvent_t end);
    
    // 内存设置
    void memsetAsync(void* dev_ptr, int value, size_t count, 
                     cudaStream_t stream = 0);
    
    // 预取内存
    void prefetchToDevice(void* unified_ptr, size_t size, 
                          int device_id = -1, cudaStream_t stream = 0);
    void prefetchToHost(void* unified_ptr, size_t size, 
                        cudaStream_t stream = 0);
    
    // 内存信息查询
    size_t getTotalMemory(int device = -1);
    size_t getFreeMemory(int device = -1);
    memoryType getmemoryType(void* ptr);
    size_t getMemorySize(void* ptr);
    
    // 清理
    void clearUnusedMemory();
    void releaseAll();
    
    // 调试信息
    void printMemoryInfo();
    void printPoolInfo();
    
private:
    // 内部辅助函数
    void checkCudaError(cudaError_t err, const char* msg);
    void* allocateAligned(size_t size, size_t alignment, memoryType type);
    void registermemoryBlock(void* ptr, size_t size, memoryType type, 
                           cudaStream_t stream = 0, size_t alignment = 0);
    void waitForasyncOperations();
};