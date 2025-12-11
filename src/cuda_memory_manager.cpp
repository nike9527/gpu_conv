// memory_manager.cu
#include "cuda_memory_manager.hpp"
#include <cassert>
#include <algorithm>

// 初始化静态成员
memoryManager* memoryManager::instance = nullptr;
std::mutex memoryManager::instance_mutex;

memoryManager::memoryManager() {
    // 初始化默认流
    streams.push_back(0); // 默认流
}

memoryManager::~memoryManager() {
    releaseAll();
}

memoryManager* memoryManager::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (!instance) {
        instance = new memoryManager();
    }
    return instance;
}

void memoryManager::checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " 
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void* memoryManager::allocate(size_t size, memoryType type, size_t alignment) {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    cudaError_t err = cudaSuccess;
    
    switch (type) {
        case memoryType::DEVICE:
            ptr = allocateAligned(size, alignment, type);
            break;
            
        case memoryType::HOST:
            err = cudaMallocHost(&ptr, size);
            checkCudaError(err, "cudaMallocHost");
            break;
            
        case memoryType::UNIFIED:
            ptr = allocateUnified(size);
            break;
            
        default:
            throw std::runtime_error("Unsupported memory type for allocate()");
    }
    
    registermemoryBlock(ptr, size, type, 0, alignment);
    return ptr;
}

void* memoryManager::allocateUnified(size_t size, unsigned int flags) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, size, flags);
    checkCudaError(err, "cudaMallocManaged");
    registermemoryBlock(ptr, size, memoryType::UNIFIED);
    return ptr;
}

void* memoryManager::allocateConstant(size_t size, const std::string& key) {
    if (size > ConstantMemoryCache::MAX_CONSTANT_SIZE) {
        throw std::runtime_error("Constant memory size exceeds limit");
    }
    
    std::lock_guard<std::mutex> lock(constant_cache.mutex);
    
    // 检查是否已缓存
    if (!key.empty() && constant_cache.cache.find(key) != constant_cache.cache.end()) {
        return constant_cache.cache[key].first;
    }
    
    // 分配新常量内存
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    checkCudaError(err, "cudaMalloc (constant)");
    
    constant_cache.used_size += size;
    
    // 加入缓存
    if (!key.empty()) {
        constant_cache.cache[key] = {ptr, size};
    }
    
    registermemoryBlock(ptr, size, memoryType::CONSTANT);
    return ptr;
}

void memoryManager::asyncCopy(void* dst, const void* src, size_t size,
                                 cudaMemcpyKind kind, cudaStream_t stream,
                                 std::function<void()> callback) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, kind, stream);
    checkCudaError(err, "cudaMemcpyAsync");
    
    // 创建事件记录
    cudaEvent_t start, end;
    err = cudaEventCreate(&start);
    err |= cudaEventCreate(&end);
    checkCudaError(err, "cudaEventCreate");
    
    cudaEventRecord(start, stream);
    
    asyncOperation op;
    op.stream = stream;
    op.start_event = start;
    op.end_event = end;
    op.bytes_transferred = size;
    op.callback = callback;
    
    switch (kind) {
        case cudaMemcpyHostToDevice:
            op.type = asyncOperation::opType::COPY_H2D;
            break;
        case cudaMemcpyDeviceToHost:
            op.type = asyncOperation::opType::COPY_D2H;
            break;
        case cudaMemcpyDeviceToDevice:
            op.type = asyncOperation::opType::COPY_D2D;
            break;
        default:
            op.type = asyncOperation::opType::COPY_H2D;
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        async_queue.push(op);
    }
}

void memoryManager::asyncHostToDevice(void* dev_ptr, const void* host_ptr,
                                         size_t size, cudaStream_t stream) {
    asyncCopy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice, stream);
}

void memoryManager::asyncDeviceToHost(void* host_ptr, const void* dev_ptr,
                                         size_t size, cudaStream_t stream) {
    asyncCopy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost, stream);
}

cudaStream_t memoryManager::createStream(bool non_blocking) {
    cudaStream_t stream;
    unsigned int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
    checkCudaError(err, "cudaStreamCreateWithFlags");
    
    streams.push_back(stream);
    return stream;
}

void memoryManager::synchronizeStream(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    checkCudaError(err, "cudaStreamSynchronize");
}

void memoryManager::synchronizeAll() {
    for (auto& stream : streams) {
        if (stream != 0) { // 默认流已在设备同步中处理
            cudaStreamSynchronize(stream);
        }
    }
    cudaDeviceSynchronize();
}

void memoryManager::memsetAsync(void* dev_ptr, int value, size_t count,
                                   cudaStream_t stream) {
    cudaError_t err = cudaMemsetAsync(dev_ptr, value, count, stream);
    checkCudaError(err, "cudaMemsetAsync");
}

void memoryManager::prefetchToDevice(void* unified_ptr, size_t size,
                                        int device_id, cudaStream_t stream) {
    cudaError_t err;
    if (device_id >= 0) {
        err = cudaMemPrefetchAsync(unified_ptr, size, device_id, stream);
    } else {
        int current_device;
        cudaGetDevice(&current_device);
        err = cudaMemPrefetchAsync(unified_ptr, size, current_device, stream);
    }
    checkCudaError(err, "cudaMemPrefetchAsync (to device)");
}

void memoryManager::free(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    auto it = memory_pool.find(ptr);
    if (it != memory_pool.end()) {
        memoryBlock& block = it->second;
        
        // 等待关联流的操作完成
        if (block.stream != 0) {
            synchronizeStream(block.stream);
        }
        
        // 释放内存
        cudaError_t err = cudaSuccess;
        switch (block.type) {
            case memoryType::DEVICE:
            case memoryType::CONSTANT:
                err = cudaFree(ptr);
                break;
            case memoryType::HOST:
                err = cudaFreeHost(ptr);
                break;
            case memoryType::UNIFIED:
                err = cudaFree(ptr);
                break;
            default:
                break;
        }
        
        checkCudaError(err, "cudaFree/cudaFreeHost");
        
        // 如果是常量内存，从缓存中移除
        if (block.type == memoryType::CONSTANT) {
            std::lock_guard<std::mutex> lock(constant_cache.mutex);
            for (auto it = constant_cache.cache.begin(); 
                 it != constant_cache.cache.end(); ++it) {
                if (it->second.first == ptr) {
                    constant_cache.used_size -= it->second.second;
                    constant_cache.cache.erase(it);
                    break;
                }
            }
        }
        
        memory_pool.erase(it);
    }
}

void memoryManager::registermemoryBlock(void* ptr, size_t size, 
                                           memoryType type, cudaStream_t stream,
                                           size_t alignment) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    memoryBlock block;
    block.ptr = ptr;
    block.size = size;
    block.type = type;
    block.is_allocated = true;
    block.stream = stream;
    block.alignment = alignment;
    
    memory_pool[ptr] = block;
}

void memoryManager::printPoolInfo() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    std::cout << "=== Memory Pool Info ===" << std::endl;
    std::cout << "Total blocks: " << memory_pool.size() << std::endl;
    
    size_t total_size = 0;
    for (const auto& pair : memory_pool) {
        const memoryBlock& block = pair.second;
        total_size += block.size;
        
        std::cout << "  Pointer: " << block.ptr 
                  << ", Size: " << block.size << " bytes"
                  << ", Type: ";
        
        switch (block.type) {
            case memoryType::DEVICE: std::cout << "Device"; break;
            case memoryType::HOST: std::cout << "Host"; break;
            case memoryType::UNIFIED: std::cout << "Unified"; break;
            case memoryType::CONSTANT: std::cout << "Constant"; break;
            default: std::cout << "Unknown";
        }
        
        std::cout << ", Stream: " << block.stream << std::endl;
    }
    
    std::cout << "Total allocated: " << total_size << " bytes" << std::endl;
}