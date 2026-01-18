#include "core/triple_pipeline.hpp"
#include "core/cuda_executor.hpp"
#include "core/gl_executor.hpp"
#include "frame_solt/cuda_frame_slot.hpp"
#include "frame_solt/gl_frame_slot.hpp"
using CudaPipe = triple_pipeline<cuda_frame_slot>;
using GLPipe = triple_pipeline<gl_frame_slot>;
int main(int argc, char *argv[])
{
    CudaPipe pipeline;
    cuda_executor executor;
    // producer
    if (auto *slot = pipeline.acquire_free())
    {
        slot->d_in_ = d_input;
        slot->d_out_ = d_output;
        slot->width_ = w;
        slot->height_ = h;

        executor.launch(*slot);
        pipeline.submit(*slot);
    }

    // consumer
    if (auto *done = pipeline.ready())
    {
        // GPU 结果已完成
        pipeline.release(*done);
    }
    //====================================================

    // triple_pipeline<gl_frame_slot> pipeline([]{ return gl_frame_slot(1920, 1080); });
    GLPipe pipeline;

    gl_executor executor;

    // producer
    if (auto *slot = pipeline.acquire_free())
    {
        // PBO 写入（CPU / CUDA / DMA）
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot->pbo());
        void *ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        // memcpy(...)
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

        executor.upload_and_draw(*slot);
        pipeline.submit(*slot);
    }

    // consumer
    if (auto *done = pipeline.ready())
    {
        // texture 可用于 display / encode
        pipeline.release(*done);
    }
    //====================================================
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    TriplePipeline<CudaGLFrameSlot> pipeline(
        []
        { return CudaGLFrameSlot(1920, 1080); });

    CudaGLExecutor executor(stream);

    // producer
    if (auto *slot = pipeline.acquire_free())
    {
        executor.launch(*slot);
        pipeline.submit(*slot);
    }

    // consumer
    if (auto *done = pipeline.try_collect())
    {
        // done->texture() 直接渲染到屏幕
        pipeline.release(*done);
    }
}
