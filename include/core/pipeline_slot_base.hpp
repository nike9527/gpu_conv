#pragma once
#include "cuda/cuda_stream.hpp"
#include "cuda/cuda_event.hpp"
#include "pipeline_state.hpp"
inline static std::atomic<int> global_slot_id{0};
class pipeline_slot_base
{
public:
    pipeline_slot_base() : id_(global_slot_id++) {}

    pipeline_state get_state() const noexcept { return state_; }
    void mark_inflight() noexcept { state_ = pipeline_state::INFLIGHT; }
    void mark_completed() noexcept { state_ = pipeline_state::COMPLETED; }
    void mark_free() noexcept { state_ = pipeline_state::FREE; }

    bool is_free() const noexcept { return state_ == pipeline_state::FREE; }
    bool is_inflight() const noexcept { return state_ == pipeline_state::INFLIGHT; }
    bool is_completed() const noexcept { return state_ == pipeline_state::COMPLETED; }

    cudaStream_t get_stream() const noexcept { return stream_.get(); }
    cudaEvent_t get_event() const noexcept { return event_.get(); }

private:
    int id_{0};
    cuda_event event_;
    cuda_stream stream_{cudaStreamNonBlocking};
    pipeline_state state_{pipeline_state::FREE};
};
