#pragma once

class cuda_mem_manage
{
private:
    float *input, *output, *kernel;
    int size, k_size;
public:
    cuda_mem_manage(int size, int k_size);
    ~cuda_mem_manage();
    void cuda_mem_copy_output(float *h_data);
    void cuda_mem_copy_input(float *h_data);
    void cuda_mem_copy_kernel(float *h_data);
};