#pragma once
#include "frame_solt/gl_frame_slot.hpp"

gl_frame_slot::gl_frame_slot(int width, int height) : width_(width), height_(height)
{
    // ---------- PBO ----------
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width_ * height_ * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // ---------- Texture ----------
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}
void gl_frame_slot::mark_submit()
{
    done_.record(stream_);
    state_.store(frame_state::INFLIGHT);

    // if (fence_)
    //     glDeleteSync(fence_);
    // fence_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    // state_ = frame_state::INFLIGHT;
}
bool gl_frame_slot::is_ready() const
{
    if (!done_.query())
        return false;
    if (fence_)
        glDeleteSync(fence_);
    if (!fence_)
    {
        fence_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        return false;
    }

    GLenum r = glClientWaitSync(fence_, 0, 0);
    return r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED;

    // if (!fence_)
    //     return false;
    // GLenum r = glClientWaitSync(fence_, 0,0);
    // return r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED;
}
