#pragma once
/**
 * 状态机逻辑
 */
#include "frame_pipeline.hpp"
template <typename T>
FramePipeline<T>::FramePipeline(size_t frame_elements, int buffers)
{
    if (buffers < 2)
        throw std::invalid_argument("FramePipeline requires >=2 buffers");

    buffers_.reserve(buffers);
    for (int i = 0; i < buffers; ++i)
        buffers_.emplace_back(frame_elements);
}

// ===== acquire =====
template <typename T>
stream_buffer<T> &FramePipeline<T>::acquire()
{
    const size_t N = buffers_.size();

    for (size_t i = 0; i < N; ++i)
    {
        auto &buf = buffers_[write_cursor_];
        write_cursor_ = (write_cursor_ + 1) % N;

        if (buf.state() == buffer_state::FREE)
            return buf;
    }

    throw std::runtime_error("FramePipeline: no FREE buffer");
}

// ===== submit =====
template <typename T>
void FramePipeline<T>::submit(stream_buffer<T> &buf)
{
    CUDA_CHECK(cudaEventRecord(buf.event(), buf.stream()));
    buf.mark_inflight();
}

// ===== try_fetch =====
template <typename T>
stream_buffer<T> *FramePipeline<T>::try_fetch()
{
    for (auto &buf : buffers_)
    {
        if (buf.state() == buffer_state::INFLIGHT)
        {
            if (cudaEventQuery(buf.event()) == cudaSuccess)
            {
                buf.mark_completed();
                return &buf;
            }
        }
    }
    return nullptr;
}

// ===== release =====
template <typename T>
void FramePipeline<T>::release(stream_buffer<T> &buf)
{
    if (buf.state() != buffer_state::COMPLETED)
        throw std::logic_error("release on non-completed buffer");

    buf.mark_free();
}
