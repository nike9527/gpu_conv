#pragma once
#include "interop/cuda_gl_frame_slot.hpp"

class CudaGLExecutor
{
public:
    explicit CudaGLExecutor(cudaStream_t stream)
        : stream_(stream) {}

    void launch(CudaGLFrameSlot &slot);

private:
    cudaStream_t stream_;
};
