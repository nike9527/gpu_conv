#include "cuda_mem_manage.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
cuda_mem_manage::cuda_mem_manage(int count, int kSize):size(count), k_size(kSize){
    CHECK_CUDA_ERROR(cudaMalloc(&input, count));
    CHECK_CUDA_ERROR(cudaMalloc(&output, count));
    CHECK_CUDA_ERROR(cudaMalloc(&kernel, kSize));
}

cuda_mem_manage::~cuda_mem_manage(){
    CHECK_CUDA_ERROR(cudaFree(input));
    CHECK_CUDA_ERROR(cudaFree(output));
    CHECK_CUDA_ERROR(cudaFree(kernel));
}
void cuda_mem_manage::cuda_mem_copy_output(float *h_data){
    CHECK_CUDA_ERROR(cudaMemcpy(output, h_data, size, cudaMemcpyDeviceToHost));
}
void cuda_mem_manage::cuda_mem_copy_input(float *h_data){
   CHECK_CUDA_ERROR(cudaMemcpy(input, h_data, size, cudaMemcpyHostToDevice));
}
void cuda_mem_manage::cuda_mem_copy_kernel(float *h_data){
  CHECK_CUDA_ERROR(cudaMemcpy(kernel, h_data, size, cudaMemcpyHostToDevice));
}