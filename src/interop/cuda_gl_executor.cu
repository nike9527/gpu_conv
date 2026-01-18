#include "interop/cuda_gl_executor.hpp"
#include "kernels/kernels.cuh"

void CudaGLExecutor::launch(CudaGLFrameSlot& slot)
{
    slot.attach_stream(stream_);

    // ---------- CUDA map ----------
    slot.map_cuda(stream_);

    // ---------- CUDA kernel ----------
    dim3 block(16, 16);
    dim3 grid(
        (slot.width() + block.x - 1) / block.x,
        (slot.height() + block.y - 1) / block.y
    );

    launchGaussianRGBA(
        static_cast<uchar4*>(slot.device_ptr()),
        slot.width(),
        slot.height(),
        stream_
    );

    slot.unmap_cuda(stream_);

    // ---------- GL upload ----------
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot.pbo());
    glBindTexture(GL_TEXTURE_2D, slot.texture());

    glTexSubImage2D(GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    slot.width(),
                    slot.height(),
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    nullptr);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // ---------- mark ----------
    slot.mark_submitted();
}
