#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glew.h>
#include <stdexcept>

class frame_slot_resource
{
    enum class State
    {
        Free,
        InFlight,
        Ready
    };

    State state = State::Free;

    // ---------- CUDA ----------
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;

    // ---------- OpenGL ----------
    GLuint pbo = 0;
    cudaGraphicsResource *cuda_pbo = nullptr;

    // ---------- mapped ptr ----------
    uchar4 *d_ptr = nullptr;
    size_t size = 0;

    int width = 0;
    int height = 0;

    void init(int w, int h);
    void acquire_cuda();

    void release_cuda();

    bool poll_ready();

    void reset();

    void destroy();
};
