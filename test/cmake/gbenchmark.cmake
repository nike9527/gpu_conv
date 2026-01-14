include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://gitee.com/yunfeiliu/benchmark.git
    GIT_TAG main
)

# 关闭 benchmark 自带的测试
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)
