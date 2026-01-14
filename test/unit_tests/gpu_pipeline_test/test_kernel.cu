#include "test_kernel.cuh"
#include "cstdio"
__global__ void dummy_kernel(float *out)
{
    if (threadIdx.x == 0){
        out[0] = 1.0f;
    }
}
void launch_dummy(stream_buffer<float> &buf)
{
    dummy_kernel<<<1, 1, 0, buf.stream()>>>(buf.d_out());
}