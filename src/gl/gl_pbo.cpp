#include "gl/gl_pbo.hpp"

GLPBO::GLPBO(int width, int height)
    : width_(width), height_(height)
{
    size_bytes_ = static_cast<size_t>(width_) *
                  static_cast<size_t>(height_) *
                  sizeof(uchar4);
    try
    {
        // 1. glGenBuffers函数生成多个未使用的缓冲区对象名称,缓冲区对象是在第一次绑定时创建的
        glGenBuffers(1, &pbo_);
        if (!pbo_)
            throw std::runtime_error("Failed to create OpenGL PBO");
        /**
         * @brief  缓冲区绑定函数 没有分配内存
         * buffer 绑定到某个用途(GL_PIXEL_UNPACK_BUFFER)
         */
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        /**
         * @brief GPU 显存中分配buffer 内存
         * 复制数据到缓冲区 GPU上分配内存并将数据填充到当前绑定的缓冲区对象
         * nullptr表示不初始化数据
         * GL_DYNAMIC_DRAW 会频繁更新
         * 数据 在 GPU 显存
         * OpenGL 拥有这块内存的控制权
         */
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size_bytes_, nullptr, GL_DYNAMIC_DRAW);
        // 解绑当前绑定的缓冲区
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        /**
         * @brief 注册OpenGL缓冲区
         *  CUDA和OpenGL之间共享缓冲区数据 不拷贝数据 不重新分配
         *  OpenGL Buffer（如 VBO / PBO）注册成 CUDA 可访问的资源，
         *  建立 CUDA ↔ OpenGL 的“资源映射关系，从而实现 GPU ↔ GPU 零拷贝数据共享 还不能访问
         *  需要通过cudaGraphicsMapResources函数调用，将图形资源映射到CUDA地址空间才能访问
         *  cudaGraphicsRegisterFlagsNone	默认，读写
         *  cudaGraphicsRegisterFlagsReadOnly	CUDA 只读
         *  cudaGraphicsRegisterFlagsWriteDiscard	CUDA 只写（最常用）
         *  cudaGraphicsRegisterFlagsSurfaceLoadStore	用作 surface
         */
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_res_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard);
        if (err != cudaSuccess)
        {
            release();
            throw std::runtime_error("cudaGraphicsGLRegisterBuffer failed");
        }
    }
    catch (...)
    {
        release();
        throw;
    }
}

GLPBO::~GLPBO()
{
    release();
}

void GLPBO::release()
{
    if (cuda_res_)
    {
        /**
         * @brief 注销图形互操作资源，释放相关CUDA资源
         * 解除 CUDA 与 OpenGL 的资源绑定
         */
        cudaGraphicsUnregisterResource(cuda_res_);
        cuda_res_ = nullptr;
    }
    if (pbo_)
    {
        glDeleteBuffers(1, &pbo_);

        pbo_ = 0;
    }
    d_ptr_ = nullptr;
}

GLPBO::GLPBO(GLPBO &&other) noexcept
{
    *this = std::move(other);
}

GLPBO &GLPBO::operator=(GLPBO &&other) noexcept
{
    if (this != &other)
    {
        release();
        width_ = other.width_;
        height_ = other.height_;
        size_bytes_ = other.size_bytes_;
        pbo_ = other.pbo_;
        cuda_res_ = other.cuda_res_;
        d_ptr_ = other.d_ptr_;

        other.pbo_ = 0;
        other.cuda_res_ = nullptr;
        other.d_ptr_ = nullptr;
        other.width_ = other.height_ = size_bytes_ = 0;
    }
    return *this;
}

void GLPBO::map(cudaStream_t stream)
{
    if (!cuda_res_)
    {
        throw std::runtime_error("PBO not registered to CUDA");
    }
    /**
     * @brief 将图形API资源映射到CUDA地址空间，使CUDA能够直接访问这些资源。
     * 映射资源 GPU buffer 的“访问权”交给 CUDA
     * OpenGL不能访问  CUDA可以访问
     * 所有在cudaGraphicsMapResources()之前发出的图形API调用会在此函数返回前完成
     * cudaGraphicsMapResources成功返回意味所有的图形API调用操作全部完成，
     * 如果图形API调用操作未完成阻塞
     *  GPU 侧资源锁 + 同步点
     */
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_res_, stream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaGraphicsMapResources failed");
    }

    size_t mapped_size = 0;
    /**
     * @brief 获取一个设备指针，通过该指针可以访问映射的图形资源
     * 指向 那块 OpenGL buffer 的显存地址
     * 这是 GPU device pointer 可以直接传给 kernel
     * CUDA kernel 直接写 OpenGL 的 buffer
     */
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&d_ptr_), &mapped_size, cuda_res_);
    if (err != cudaSuccess || mapped_size < size_bytes_)
    {
        throw std::runtime_error("cudaGraphicsResourceGetMappedPointer failed");
    }
}

void GLPBO::unmap(cudaStream_t stream)
{
    if (!cuda_res_)
        return;
    /**
     * @brief 解除已映射的CUDA图形资源，让图形API可以重新安全地访问这些资源
     * 把 buffer 归还给 OpenGL
     * CUDA 不能再访问 OpenGL可以使用
     * CUDA → OpenGL 同步点
     * 同步点 + 资源所有权切换 此函数提供了同步保证，确保在流中发出的任何CUDA工作
     * 此函数使用标准的默认流语义
     * 请注意，此函数还可能返回之前异步启动时产生的错误代码
     */
    cudaError_t err = cudaGraphicsUnmapResources(1, &cuda_res_, stream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaGraphicsUnmapResources failed");
    }

    d_ptr_ = nullptr;
}