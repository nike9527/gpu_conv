triple_pipeline<float> pipeline(W *H);
while (running)
{
    auto &buf = pipeline.acquire();
    std::memcpy(buf.h_in(), input_frame, W * H * sizeof(float));
    // 异步 H2D
    cudaMemcpyAsync(buf.d_in(), buf.h_in(), W * H * sizeof(float),
                    cudaMemcpyHostToDevice, buf.stream());
    // 异步 kernel
    launchFilter(buf.d_in(), buf.d_out(), W, H, mem_type::GLOBAL, filterObj,
                 16, 16, buf.stream());
    // 异步 D2H
    cudaMemcpyAsync(buf.h_out(), buf.d_out(), W * H * sizeof(float),
                    cudaMemcpyDeviceToHost, buf.stream());
    pipeline.submit(buf);
    // 拉取完成的 buffer
    if (auto *done = pipeline.try_fetch())
    {
        consume(done->h_out());
    }
}