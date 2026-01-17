#pragma once
#include <memory>
#include "gl/gl_pbo.hpp"
#include "cuda/cuda_memory.hpp"
#include "cuda/cuda_event.hpp"
#include "core/pipeline_state.hpp"
#include "core/pipeline_slot_base.hpp"
class gl_frame_slot
{
public:
    cuda_stream stream{cudaStreamNonBlocking};
    cuda_event event;
    std::unique_ptr<GLPBO> pbo;

private:
    pipeline_state state{pipeline_state::FREE};

public:
    explicit gl_frame_slot() = default;
    bool is_free() const noexcept { return state == pipeline_state::FREE; }
    bool is_inflight() const noexcept { return state == pipeline_state::INFLIGHT; }
    bool is_completed() const noexcept { return state == pipeline_state::COMPLETED; }

    void mark_inflight() noexcept { state = pipeline_state::INFLIGHT; }
    void mark_completed() noexcept { state = pipeline_state::COMPLETED; }
    void mark_free() noexcept { state = pipeline_state::FREE; }

    cudaStream_t get_stream() const noexcept { return stream.get(); }
    cudaEvent_t get_event() const noexcept { return event.get(); }
};
