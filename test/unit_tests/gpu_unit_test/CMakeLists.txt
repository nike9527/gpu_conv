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

include(GoogleTest)
gtest_discover_tests(unit_gpu_tests)
