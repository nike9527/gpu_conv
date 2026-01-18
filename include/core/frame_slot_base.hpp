#pragma once
#include "frame_state.hpp"
#include <atomic>
inline static std::atomic<int> global_slot_id{0};
class frame_slot_base
{
public:
    frame_slot_base();
    frame_state get_state() const noexcept;
    void set_state(frame_state s) noexcept;
    virtual ~frame_slot_base() = default;
    virtual void mark_submit() = 0;
    virtual bool is_ready() const = 0;

protected:
    int id_{0};
    std::atomic<frame_state> state_{frame_state::FREE};
};
