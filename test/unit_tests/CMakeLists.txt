add_executable(unit_sobel_test
    test_sobel.cu
)

target_link_libraries(unit_sobel_test
    PRIVATE
    cuda_conv_lib
    GTest::gtest
    GTest::gtest_main
)

add_test(
    NAME unit_sobel
    COMMAND unit_sobel_test
)
