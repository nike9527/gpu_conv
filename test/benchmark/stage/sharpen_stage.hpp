#pragma once
#include "pipeline_stage.hpp"
#include "kernels/kernels.cuh"

class sharpen_stage final : public pipeline_stage<float>
{
public:
    sharpen_stage(int w, int h, int kSize)
        : width_(w), height_(h), kSize_(kSize)
    {
        block_ = dim3(16, 16);
        grid_ = dim3(
            (width_ + block_.x - 1) / block_.x,
            (height_ + block_.y - 1) / block_.y);

        shared_bytes_ =
            (block_.x + (2 * kSize_ / 2)) *
            (block_.y + (2 * kSize_ / 2)) *
            sizeof(float);
    }

    void enqueue(stream_buffer<float> &buf) override
    {
        sharpenConvolutionWithShared<<<
            grid_, block_, shared_bytes_, buf.stream()>>>(
            buf.d_in(), buf.d_out(),
            width_, height_, kSize_);
    }

private:
    int width_;
    int height_;
    int kSize_;

    dim3 block_;
    dim3 grid_;
    size_t shared_bytes_;
};
