#include "core/gl_executor.hpp"
#include <GL/glew.h>

void gl_executor::upload_and_draw(gl_frame_slot &slot)
{
    // ---------- 上传 PBO -> Texture ----------
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot.pbo());
    glBindTexture(GL_TEXTURE_2D, slot.texture());

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, slot.width(), slot.height(), GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // ---------- draw（示意） ----------
    // 真实项目中：bind shader / vao / draw quad
    // glDrawArrays(...)

    slot.mark_submit();
}
