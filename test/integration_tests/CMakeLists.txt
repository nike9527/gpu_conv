add_executable(integration_tests
    test_pipeline.cpp
)

target_link_libraries(integration_tests
    PRIVATE
    gconv
    GTest::gtest
    GTest::gtest_main
)
copy_runtime_dlls(conv_benchmark)
copy_runtime_dlls(sample_sobel)
include(GoogleTest)
if (WIN32)
gtest_discover_tests(integration_tests DISCOVERY_MODE POST_BUILD)
else()
    gtest_discover_tests(unit_cpu_tests)
endif()