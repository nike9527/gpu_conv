#pragma once
#include "frame_solt/cuda_frame_slot.hpp"

/**
 * @brief CUDA executor
 *
 *  - 唯一允许 launch kernel 的地方
 *  - 不管理 slot 生命周期
 */
class cuda_executor
{
public:
    void launch(cuda_frame_slot &slot, int width, int height);
};
