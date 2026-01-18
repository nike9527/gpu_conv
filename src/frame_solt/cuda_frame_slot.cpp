#include "frame_solt/cuda_frame_slot.hpp"

cuda_frame_slot::cuda_frame_slot(int width, int height) : width_(width), height_(height)
{
}
cuda_frame_slot::~cuda_frame_slot()
{
}
void cuda_frame_slot::mark_submit()
{
    cudaEventRecord(done_, stream_);
    state_.store(frame_state::INFLIGHT, std::memory_order_release);
}
bool cuda_frame_slot::is_ready() const
{
    return cudaEventQuery(done_) == cudaSuccess;
}
