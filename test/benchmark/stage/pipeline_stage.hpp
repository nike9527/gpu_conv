#pragma once
#include "core/stream_buffer.hpp"

template <typename T>
struct pipeline_stage
{
    virtual ~pipeline_stage() = default;
    virtual void enqueue(stream_buffer<T> &buf) = 0;
};
