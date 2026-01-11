#include "pipeline_runner.hpp"
#include "triple_pipeline.hpp"
#include "gpu_conv_kernels.hpp"

int main()
{
    constexpr int W = 1920;
    constexpr int H = 1080;
    constexpr int FRAMES = 200;
    constexpr size_t ELEMS = W * H;
    constexpr size_t BYTES = ELEMS * sizeof(float);

    auto launch = [&](stream_buffer<float>& buf)
    {
        fill_input(buf.h_in(), W, H);

        cudaMemcpyAsync(
            buf.d_in(), buf.h_in(),
            BYTES,
            cudaMemcpyHostToDevice,
            buf.stream());

        launchGaussian(
            buf.d_in(), buf.d_out(),
            W, H, 3,
            buf.stream());

        cudaMemcpyAsync(
            buf.h_out(), buf.d_out(),
            BYTES,
            cudaMemcpyDeviceToHost,
            buf.stream());
    };

    auto consume = [&](stream_buffer<float>& buf)
    {
        consume_output(buf.h_out(), W, H);
    };

    {
        single_pipeline sp(ELEMS);
        double ms = run_pipeline_bench(sp, FRAMES, BYTES, launch, consume);
        printf("[Single]  %.3f ms/frame\n", ms);
    }

    {
        triple_pipeline<float, 3> tp(ELEMS);
        double ms = run_pipeline_bench(tp, FRAMES, BYTES, launch, consume);
        printf("[Triple]  %.3f ms/frame\n", ms);
    }
}
