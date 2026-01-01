add_executable(unit_gpu_tests
    test_conv_gpu.cu
)

set_target_properties(unit_gpu_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(unit_gpu_tests
    PRIVATE
    gconv
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)
copy_runtime_dlls(unit_cpu_tests)
include(GoogleTest)
if (WIN32)
gtest_discover_tests(unit_gpu_tests DISCOVERY_MODE POST_BUILD)
else()
    gtest_discover_tests(unit_cpu_tests)
endif()