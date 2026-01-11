#pragma once
#include <chrono>
#include <cuda_runtime.h>

template <typename Pipeline, typename LaunchFn, typename ConsumeFn>
double run_pipeline_bench(
    Pipeline &pipeline,
    int frames,
    size_t bytes,
    LaunchFn &&launch,
    ConsumeFn &&consume)
{
    // warmup
    for (int i = 0; i < 10; ++i)
    {
        auto &buf = pipeline.acquire();
        launch(buf);
        pipeline.submit(buf);
    }

    while (pipeline.inflight() > 0)
    {
        if (auto *b = pipeline.try_fetch())
        {
            consume(*b);
            pipeline.release(*b);
        }
    }

    cudaDeviceSynchronize();

    // steady
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < frames; ++i)
    {
        auto &buf = pipeline.acquire();
        launch(buf);
        pipeline.submit(buf);

        if (auto *b = pipeline.try_fetch())
        {
            consume(*b);
            pipeline.release(*b);
        }
    }

    while (pipeline.inflight() > 0)
    {
        if (auto *b = pipeline.try_fetch())
        {
            consume(*b);
            pipeline.release(*b);
        }
    }

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    return ms / frames;
}
