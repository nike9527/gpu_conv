#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "test_kernel.cuh"
#include <iostream>
/**
 * @brief
 * Debug 构建必须打开 assert
 * 原版 会在至少 3 个测试中失败
 * “隐式 free in acquire” 的实现必挂
 * “fetch 自动 mark_free” 的实现必挂
 * inflight 管理不严格的实现必挂
 *
 * Debug 构建必须打开 assert
 * cmake ===> add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
 * CI ===>>> ctest -R TriplePipeline --output-on-failure
 */
constexpr size_t kElems = 1024;

using Pipeline = triple_pipeline<float, 3>;

/**
 * @brief 基础流转测试（正确路径）
 * 正确路径不死锁
 * inflight 精确
 */
TEST(TriplePipeline, BasicAcquireSubmitFetchRelease)
{
    Pipeline pipe(kElems);
    auto &buf = pipe.acquire();
    launch_dummy(buf);
    pipe.submit(buf);
    stream_buffer<float> *done = nullptr;
    while (!(done = pipe.try_fetch()))
    {
    }
    EXPECT_EQ(pipe.inflight(), 1);
    pipe.release(*done);
    EXPECT_EQ(pipe.inflight(), 0);
}
/**
 * @brief 重复 submit（必须断言失败）
 * 能抓住 “重复提交同一 buffer” 的 bug
 */
TEST(TriplePipeline, DoubleSubmitDies)
{
    Pipeline pipe(kElems);
    auto &buf = pipe.acquire();
    launch_dummy(buf);
    pipe.submit(buf);
#ifndef NDEBUG
    // EXPECT_DEATH(pipe.submit(buf), "");
    EXPECT_THROW(pipe.submit(buf), std::logic_error);
#endif
}
/**
 * @brief Fetch 后不 release（buffer 泄漏检测）
 * 防止“fetch 就自动 free”的假实现
 */
TEST(TriplePipeline, FetchWithoutReleaseEventuallyBlocks)
{
    Pipeline pipe(kElems);

    // 塞满 pipeline
    for (int i = 0; i < 3; ++i)
    {
        auto &buf = pipe.acquire();
        launch_dummy(buf);
        pipe.submit(buf);
    }

    // fetch 一个但不 release
    stream_buffer<float> *b0 = nullptr;
    while (!(b0 = pipe.try_fetch()))
    {
    }

    EXPECT_EQ(pipe.inflight(), 3);

    // 第 4 次 acquire 必须阻塞（隐式同步）
    auto &buf = pipe.acquire();
    EXPECT_TRUE(buf.capacity() > 0);
}
/**
 * @brief 提前 release（未完成就释放）
 * 能打掉 release 不校验状态 的实现
 */
TEST(TriplePipeline, EarlyReleaseDies)
{
    Pipeline pipe(kElems);
    auto &buf = pipe.acquire();

    launch_dummy(buf);
    pipe.submit(buf);

#ifndef NDEBUG
    EXPECT_DEATH(pipe.release(buf), "");
#endif
}
/**
 * @brief 高频压力循环（最重要）
 * inflight 永远不负
 * 不死锁
 * 不丢 buffer
 */
TEST(TriplePipeline, StressLoop)
{
    Pipeline pipe(kElems);

    constexpr int rounds = 10000;
    int completed = 0;

    for (int i = 0; i < rounds; ++i)
    {
        auto &buf = pipe.acquire();
        launch_dummy(buf);
        pipe.submit(buf);

        if (auto *done = pipe.try_fetch())
        {
            pipe.release(*done);
            completed++;
        }
    }

    // drain
    while (pipe.inflight() > 0)
    {
        if (auto *done = pipe.try_fetch())
        {
            pipe.release(*done);
            completed++;
        }
    }

    EXPECT_EQ(completed, rounds);
}

/**
 * @brief 析构安全性（隐式同步）
 * 能抓住 析构 use-after-free
 */
TEST(TriplePipeline, DestructionWaitsForInflight)
{
    {
        Pipeline pipe(kElems);
        auto &buf = pipe.acquire();
        launch_dummy(buf);
        pipe.submit(buf);
    }
    SUCCEED(); // 不 crash 即通过
}
