__global__ void dummy_kernel(float *out)
{
    if (threadIdx.x == 0)
        out[0] = 1.0f;
}
