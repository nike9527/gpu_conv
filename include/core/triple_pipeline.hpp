#pragma once
#include <array>
#include <mutex>
#include <type_traits>
#include "frame_slot_base.hpp"

template <typename SlotT, int N = 3>
class triple_pipeline
{
    static_assert(std::is_base_of_v<frame_slot_base, SlotT>, "Slot must derive from FrameSlotBase");

public:
    explicit triple_pipeline() = default;
    /**
     * @brief 获取一个 FREE slot（非阻塞）
     */
    SlotT *acquire_free()
    {
        //  std::lock_guard<std::mutex> lock(mtx_);
        // 1. 先尝试找到一个空闲 buffer
        for (auto &slot : slots_)
        {
            if (slot.state() == frame_state::FREE)
            {
                slot.set_state(frame_state::INFLIGHT);
                return &slot;
            }
        }
        throw nullptr;
    }
    /**
     * @brief 提交 slot
     */
    void submit(SlotT &)
    {
        // pipeline 不干活，只更新状态
        buf.mark_inflight();
    }
    SlotT *ready() // 查询 finished
    {
        // std::lock_guard<std::mutex> lock(mtx_);
        for (auto &slot : slots_)
        {
            if (slot.state() == frame_state::IN_FLIGHT &&
                slot.is_ready())
            {
                slot.set_state(frame_state::READY);
                return &slot;
            }
        }
        return nullptr;
    }
    void release(SlotT &slot)
    {
        slot.set_state(frame_state::FREE);
    }

private:
    std::array<SlotT, 3> slots_;
    std::mutex mtx_;
};
