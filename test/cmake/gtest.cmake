include(FetchContent)
# 确保使用动态运行时库
if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
# 下载并获取 GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://gitee.com/mirrors/googletest.git
    GIT_TAG v1.17.0
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

if (NOT TARGET gtest OR NOT TARGET gtest_main)
    message(FATAL_ERROR "GoogleTest FetchContent failed")
endif()

# 确保GTest使用正确的运行时库
if(MSVC)
    # 设置GTest目标属性
    foreach(target IN ITEMS gtest gtest_main gmock gmock_main)
        if(TARGET ${target})
            set_target_properties(${target} PROPERTIES
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
            )
        endif()
    endforeach()
endif()

message(STATUS "GoogleTest found: ${GTest_VERSION}")
message(STATUS "GoogleTest include dir: ${GTEST_INCLUDE_DIRS}")
message(STATUS "GoogleTest libraries: ${GTEST_LIBRARIES}")

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# 验证
if(TARGET gtest)
    message(STATUS "GoogleTest downloaded and configured")
    include(GoogleTest)
else()
    message(WARNING "GoogleTest setup completed but no gtest target found")
endif()

