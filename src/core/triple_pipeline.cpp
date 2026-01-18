#include "frame_pipeline.hpp"
#include "cuda/cuda_exception.hpp"
frame_pipeline::frame_pipeline(int slots)
{
}

frame_slot *frame_pipeline::acquire()
{
    for (auto &slot : slots_)
    {
        if (slot.state_ == frame_state::FREE)
        {
            slot.state_ = frame_state::INFLIGHT;
            return &slot;
        }
    }
    return nullptr;
}
void frame_pipeline::submit(frame_slot &slot)
{
    if (!(slot.state_ == frame_state::FREE))
    {
#ifndef NDEBUG
        throw std::logic_error("Cannot submit non-free buffer.");
#else
        return;
#endif
    }
    CUDA_CHECK(cudaEventRecord(slot.done_.get(), slot.stream_.get()));
    // pipeline 不干活，只更新状态
    slot.state_ = frame_state::INFLIGHT;
}
frame_slot *frame_pipeline::is_ready()
{
    for (auto &slot : slots_)
    {
        if (slot.state_ == frame_state::FREE)
        {
        }
        if (slot.state() == slot_state::INFLIGHT && slot.is_ready())
        {
            slot.set_state(slot_state::COMPLETED);
            return &slot;
        }
    }
    return nullptr;
}
