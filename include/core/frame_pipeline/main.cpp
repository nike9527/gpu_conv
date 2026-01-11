#include "frame_pipeline.hpp"
int main()
{
    int width = 100;
    int height = 100;
    int bytes = 100;
    FramePipeline<float> pipeline(width * height);

    auto &buf = pipeline.acquire();

    // enqueue work
    cudaMemcpyAsync(buf.d_in(), buf.h_in(), bytes,
                    cudaMemcpyHostToDevice, buf.stream());

    stage.enqueue(buf);

    cudaMemcpyAsync(buf.h_out(), buf.d_out(), bytes,
                    cudaMemcpyDeviceToHost, buf.stream());

    pipeline.submit(buf);

    // elsewhere
    if (auto *done = pipeline.try_fetch())
    {
        consume(done->h_out());
        pipeline.release(*done);
    }
}