#pragma once
#include "cuda_exception.hpp"
#include <cstddef>
#include <utility>

template <typename T>
class cuda_memory
{
public:
    cuda_memory() = default;

    explicit cuda_memory(size_t count) : count_(count)
    {
        CUDA_CHECK(cudaMalloc(&ptr_, sizeof(T) * count_));
    }
    ~cuda_memory() noexcept { release(); }
    // 禁拷贝
    cuda_memory(const cuda_memory &) = delete;
    cuda_memory &operator=(const cuda_memory &) = delete;
    // 允许 move
    cuda_memory(cuda_memory &&other) noexcept
    {
        move_from(std::move(other));
    }
    cuda_memory &operator=(cuda_memory &&other) noexcept
    {
        if (this != &other)
        {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    T *data() noexcept { return ptr_; }
    const T *data() const noexcept { return ptr_; }

    size_t size() const noexcept { return count_; }

    void copy_from_host_async(const T *h, size_t count, cudaStream_t stream = 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, h, sizeof(T) * count, cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host_async(T *h, size_t count, cudaStream_t stream = 0) const
    {
        CUDA_CHECK(cudaMemcpyAsync(h, ptr_, sizeof(T) * count, cudaMemcpyDeviceToHost, stream));
    }
    void copy_from_host(const T *h, size_t count)
    {
        CUDA_CHECK(cudaMemcpy(ptr_, h, sizeof(T) * count, cudaMemcpyHostToDevice));
    }

    void copy_to_host(T *h, size_t count) const
    {
        CUDA_CHECK(cudaMemcpy(h, ptr_, sizeof(T) * count, cudaMemcpyDeviceToHost));
    }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;

    void release() noexcept
    {
        if (ptr_)
            cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }

    void move_from(cuda_memory &&other) noexcept
    {
        ptr_ = other.ptr_;
        count_ = other.count_;
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
};
