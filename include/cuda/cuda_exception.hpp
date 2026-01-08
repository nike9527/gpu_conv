#pragma once
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>  // 对于Driver API
namespace cuda_error
{
class cuda_exception : public std::runtime_error {
public:
    cuda_exception(cudaError_t err, const char* file, int line): std::runtime_error(format(err, file, line)),
          error_(err) {}
    
    cuda_exception(CUresult err, const char* file, int line): std::runtime_error(format(err, file, line)),driver_error_(err) {}
    
    cudaError_t get_runtime_error() const noexcept { return error_; }
    CUresult get_driver_error() const noexcept { return driver_error_; }

private:
    static std::string format(cudaError_t err, const char* file, int line) {
        char buffer[2048];
        // snprintf(buffer, sizeof(buffer),
        //             "[CUDA Runtime Error] at %s:%d\n  Code: %d (%s)\n  Description: %s",
        //             file, line,static_cast<int>(err),cudaGetErrorName(err),cudaGetErrorString(err)
        //         );
        printf("[CUDA Runtime Error] at %s:%d\n  Code: %d (%s)\n  Description: %s",
            file, line,static_cast<int>(err),cudaGetErrorName(err),cudaGetErrorString(err)
        );
        return buffer;
    }
    
    static std::string format(CUresult err, const char* file, int line) {
        const char* name = nullptr;
        const char* desc = nullptr;
        
        cuGetErrorName(err, &name);
        cuGetErrorString(err, &desc);
        
        char buffer[512];
        // snprintf(buffer, sizeof(buffer),
        //         "[CUDA Driver Error] at %s:%d\n  Code: %d\n  Name: %s\n  Description: %s",
        //         file, line,static_cast<int>(err),name ? name : "Unknown",desc ? desc : "Unknown");
        printf("[CUDA Driver Error] at %s:%d\n  Code: %d\n  Name: %s\n  Description: %s",
            file, line,static_cast<int>(err),name ? name : "Unknown",desc ? desc : "Unknown");
        return buffer;
    }
    
    cudaError_t error_ = cudaSuccess;
    CUresult driver_error_ = CUDA_SUCCESS;
};

} // namespace name
#define CUDA_CHECK(code) \
    do { \
        cudaError_t __result = (code); \
        if (__result != cudaSuccess) { \
            throw cuda_error::cuda_exception(__result, __FILE__, __LINE__); \
        } \
    } while(0)

#define CUDA_DRIVER_CHECK(code) \
    do { \
        cudaError_t __result = (code); \
        if (__result != CUDA_SUCCESS) { \
            throw cuda_error::cuda_exception(__result, __FILE__, __LINE__); \
        } \
    } while(0)

#define CHECK_KERNEL_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError();   \
        if (err != cudaSuccess) { \
            throw cuda_error::cuda_exception(err, __FILE__, __LINE__); \
        } \
    } while(0)
// 同步错误检查
inline void sync_check(const char* file = __FILE__, int line = __LINE__) {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw cuda_error::cuda_exception(err, file, line);
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw cuda_error::cuda_exception(err, file, line);
    }
}

#define CUDA_SYNC_CHECK() cuda_error::sync_check(__FILE__, __LINE__)