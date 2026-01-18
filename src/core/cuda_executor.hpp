#pragma once
#include "core/cuda_executor.hpp"
// ================= 示例 kernel =================
__global__ void copy_kernel(const float *in,
                            float *out,
                            int width,
                            int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        out[idx] = in[idx];
    }
}
void cuda_executor::launch(cuda_frame_slot &slot, int width, int height)
{

    // === 示例：launch kernel ===
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // launchConvKernel(slot.d_in, slot.d_out, width, height, slot.stream_);

    // 告诉 slot：GPU 工作已提交
    slot.mark_submit();
    //=======================================================
    // if (!slot.d_in || !slot.d_out)
    //     return;

    // dim3 block(16, 16);
    // dim3 grid(
    //     (slot.width_ + block.x - 1) / block.x,
    //     (slot.height_ + block.y - 1) / block.y);

    // copy_kernel<<<grid, block, 0, slot.stream()>>>(slot.d_in, slot.d_out, slot.width, slot.height);

    // // 通知 slot：GPU 任务已经提交
    // slot.mark_submit();
}
