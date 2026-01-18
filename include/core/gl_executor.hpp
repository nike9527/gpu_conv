#pragma once
#include "frame_solt/gl_frame_slot.hpp"

/**
 * @brief OpenGL Executor
 *
 * 唯一职责：
 *  - 使用 slot 的 GL 资源
 *  - 执行 upload / draw
 *  - 调用 slot.mark_submitted()
 */
class GLExecutor
{
public:
    void upload_and_draw(gl_frame_slot &slot);
};
