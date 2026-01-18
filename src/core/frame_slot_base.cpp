#include "frame_slot_base.hpp"
frame_slot_base::frame_slot_base() : id_(global_slot_id++) {}

frame_state frame_slot_base::get_state() const noexcept
{
    return state_.load(std::memory_order_acquire);
}

void frame_slot_base::set_state(frame_state s) noexcept
{
    state_.store(s, std::memory_order_release);
}