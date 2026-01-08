#pragma once
#include "core/stream_buffer.hpp"
#include "core/triple_pipeline.hpp"
class frame_processor {
public:
    frame_processor(int w, int h): width_(w), height_(h),pipe_(w * h){}
    // 非阻塞提交
    void submit(const float* input);
    // 非阻塞取回
    bool try_fetch(float* output);
private:
    int bytes() const { return width_ * height_ * sizeof(float); }
    void launch_kernel(stream_buffer<float>& buf);
private:
     int width_, height_;
    triple_pipeline<float> pipe_;

};
