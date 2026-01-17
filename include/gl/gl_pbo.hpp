#pragma once
#include "gl_headers.hpp"
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
class GLPBO
{
public:
    GLPBO(int width, int height);
    ~GLPBO();

    GLPBO(const GLPBO &) = delete;
    GLPBO &operator=(const GLPBO &) = delete;

    GLPBO(GLPBO &&other) noexcept;
    GLPBO &operator=(GLPBO &&other) noexcept;

    // CUDA <-> GL
    void map(cudaStream_t stream);
    void unmap(cudaStream_t stream);

    // Accessors
    uchar4 *device_ptr() const noexcept { return d_ptr_; }
    GLuint pbo() const noexcept { return pbo_; }
    size_t size_bytes() const noexcept { return size_bytes_; }

private:
    void release();

    int width_;
    int height_;
    size_t size_bytes_;

    GLuint pbo_ = 0;
    // CUDA图形互操作中的核心资源对象，用于管理CUDA与图形API（OpenGL、Direct3D、Vulkan）之间的共享资源+
    cudaGraphicsResource *cuda_res_ = nullptr;
    uchar4 *d_ptr_ = nullptr;
};
