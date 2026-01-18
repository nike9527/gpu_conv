#pragma once
#include "runtime/frame_slot.hpp"

struct DummySlot : public FrameSlotBase
{
    bool done = false;

    void mark_submitted() override
    {
        done = false;
    }

    bool is_ready() const override
    {
        return done;
    }
};
