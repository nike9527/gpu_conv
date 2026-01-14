cmake_minimum_required(VERSION 3.18)
#---------------------------------------------
# cmake Test
#---------------------------------------------
include(CTest)
enable_testing()

#---------------------------------------------
# 测试构建选项
# ENABLE_UNIT_TESTS         是否开启单元测试
# ENABLE_INTEGRATION_TESTS  是否开启集成测试
# ENABLE_BENCHMARKS         是否开启基准测试
#---------------------------------------------
option(ENABLE_UNIT_TESTS        "Build unit tests"          ON)
option(ENABLE_INTEGRATION_TESTS "Build integration tests"   OFF)
option(ENABLE_BENCHMARKS        "Build benchmarks"          ON)

message(STATUS "  Test config:")
message(STATUS "  Unit tests        : ${ENABLE_UNIT_TESTS}")
message(STATUS "  Integration tests : ${ENABLE_INTEGRATION_TESTS}")
message(STATUS "  Benchmarks        : ${ENABLE_BENCHMARKS}")

# 确保gconv库能被正确找到
if(NOT TARGET gconv)
    message(FATAL_ERROR "gconv target not found!")
endif()

#---------------------------------------------
# 子目录调度
#---------------------------------------------
if (ENABLE_UNIT_TESTS)
    add_subdirectory(unit_tests)
endif()

if (ENABLE_INTEGRATION_TESTS)
    add_subdirectory(integration_tests)
endif()

if (ENABLE_BENCHMARKS)
    add_subdirectory(benchmark)
endif()


