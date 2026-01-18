#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glew.h>

#include "core/pipeline_slot_base.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_exception.hpp"

class CudaGLFrameSlot : public PipelineSlotBase
{
public:
    CudaGLFrameSlot(int width, int height)
        : width_(width), height_(height)
    {
        // ---------------- PBO ----------------
        glGenBuffers(1, &pbo_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     width_ * height_ * 4,
                     nullptr,
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // ---------------- Texture ----------------
        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA8,
                     width_,
                     height_,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // ---------------- CUDA interop ----------------
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
            &cuda_res_,
            pbo_,
            cudaGraphicsRegisterFlagsWriteDiscard));
    }

    ~CudaGLFrameSlot() override
    {
        if (cuda_res_)
            cudaGraphicsUnregisterResource(cuda_res_);
        if (gl_fence_)
            glDeleteSync(gl_fence_);
        glDeleteBuffers(1, &pbo_);
        glDeleteTextures(1, &texture_);
    }

    CudaGLFrameSlot(const CudaGLFrameSlot &) = delete;
    CudaGLFrameSlot &operator=(const CudaGLFrameSlot &) = delete;

    // ---------------- accessors ----------------
    GLuint pbo() const noexcept { return pbo_; }
    GLuint texture() const noexcept { return texture_; }

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }

    // ---------------- CUDA mapping ----------------
    void map_cuda(cudaStream_t stream)
    {
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_res_, stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            &device_ptr_,
            &mapped_bytes_,
            cuda_res_));
    }

    void unmap_cuda(cudaStream_t stream)
    {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_res_, stream));
    }

    void *device_ptr() const noexcept { return device_ptr_; }

    // ---------------- pipeline state ----------------
    void mark_submitted() override
    {
        cuda_done_.record(stream_);
        set_state(SlotState::IN_FLIGHT);
    }

    bool is_ready() override
    {
        if (!cuda_done_.query())
            return false;

        if (!gl_fence_)
        {
            gl_fence_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            return false;
        }

        GLenum r = glClientWaitSync(gl_fence_, 0, 0);
        return r == GL_ALREADY_SIGNALED ||
               r == GL_CONDITION_SATISFIED;
    }

    void attach_stream(cudaStream_t s) { stream_ = s; }

private:
    int width_;
    int height_;

    GLuint pbo_ = 0;
    GLuint texture_ = 0;

    cudaGraphicsResource *cuda_res_ = nullptr;
    void *device_ptr_ = nullptr;
    size_t mapped_bytes_ = 0;

    cudaStream_t stream_ = nullptr;
    cudaEventWrapper cuda_done_;
    GLsync gl_fence_ = nullptr;
};
