#include "stream_buffer.hpp"
#include "cuda/cuda_memory.hpp"
struct single_pipeline
{
    stream_buffer<float> buf;

    explicit single_pipeline(size_t elems) : buf(elems) {}

    stream_buffer<float> &acquire() { return buf; }

    void submit(stream_buffer<float> &b)
    {
        CUDA_CHECK(cudaEventRecord(b.event(), b.stream()));
        b.mark_busy();
    }

    stream_buffer<float> *try_fetch()
    {
        if (!buf.busy())
            return nullptr;
        if (cudaEventQuery(buf.event()) != cudaSuccess)
            return nullptr;
        return &buf;
    }

    void release(stream_buffer<float> &b)
    {
        b.mark_free();
    }

    int inflight() const { return buf.busy() ? 1 : 0; }
};
