#include "frame_processor.hpp"
// 非阻塞提交
void frame_processor::submit(const float* input){
    auto& buf = pipe_.acquire();
    std::memcpy(buf.h_in(), input, bytes());
    cudaMemcpyAsync(buf.d_in(), buf.h_in(), bytes(),cudaMemcpyHostToDevice, buf.stream());
    launch_kernel(buf);   // 只负责 kernel
    cudaMemcpyAsync(buf.h_out(), buf.d_out(), bytes(),cudaMemcpyDeviceToHost, buf.stream());
    pipe_.submit(buf);
}
// 非阻塞取回
bool frame_processor::try_fetch(float* output){
    if (auto* done = pipe_.try_fetch()) {
        std::memcpy(output, done->h_out(), bytes());
        return true;
    }
    return false;
}