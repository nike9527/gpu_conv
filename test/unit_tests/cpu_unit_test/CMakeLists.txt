
#---------------------------------------------
# 添加可执行文件
#---------------------------------------------
add_executable(unit_cpu_tests test_conv_cpu.cpp)
#---------------------------------------------
# 链接库
#---------------------------------------------
target_link_libraries(unit_cpu_tests
    PRIVATE
    gconv
    GTest::gtest 
    GTest::gtest_main  # 链接 gtest_main 提供 main 函数
)
# 关键：设置运行目录
# set_target_properties(unit_cpu_tests PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

copy_runtime_dlls(unit_cpu_tests)
# set_target_properties(unit_cpu_tests PROPERTIES
#     MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>"
# )
#---------------------------------------------
#自动发现 GoogleTest 测试用例
#---------------------------------------------
include(GoogleTest)
if (WIN32)
gtest_discover_tests(unit_cpu_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    DISCOVERY_MODE POST_BUILD
)
else()
    gtest_discover_tests(unit_cpu_tests)
endif()
#---------------------------------------------
# 添加测试
#---------------------------------------------
# add_test(NAME unit_conv2dCpu 
#         COMMAND $<TARGET_FILE:unit_cpu_tests> 
# )
