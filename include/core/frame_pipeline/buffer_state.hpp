#pragma once
#include <cstdint>

enum class buffer_state : uint8_t
{
    FREE,     // 可 acquire：GPU 不使用 == pipeline: 不占用 不持有 ==  CPU: 不可读
    INFLIGHT, // 已 submit：GPU 可能正在使用  == pipeline: 持有  == CPU: 不可读
    COMPLETED // GPU: 已完成，不使用 == pipeline: 仍持有（等待 CPU 消费） == CPU:可读
};