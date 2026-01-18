#pragma once
#include "gl/gl_pbo.hpp"
#include "cuda/cuda_memory.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_stream.hpp"
#include "core/frame_slot_base.hpp"
class gl_frame_slot : public frame_slot_base
{
public:
    cuda_event done_;
    cuda_stream stream_{cudaStreamNonBlocking};

    int width_ = 0;
    int height_ = 0;
    int frame_id = -1;
    cuda_memory<float> d_input;
    cuda_memory<float> d_output;
    GLuint pbo_ = 0;
    GLuint texture_ = 0;
    GLsync fence_ = nullptr;

    gl_frame_slot(int width, int height);
    
    gl_frame_slot(const gl_frame_slot &) = delete;
    gl_frame_slot &operator=(const gl_frame_slot &) = delete;

    virtual void mark_submit() override;
    virtual bool is_ready() const override;

private:
    virtual ~gl_frame_slot() override
    {
        if (fence_)
            glDeleteSync(fence_);
        glDeleteBuffers(1, &pbo_);
        glDeleteTextures(1, &texture_);
    }
};
