add_executable(integration_pipeline_test
    test_pipeline.cu
)

target_link_libraries(integration_pipeline_test
    PRIVATE
    cuda_conv_lib
    GTest::gtest
    GTest::gtest_main
)

add_test(
    NAME integration_pipeline
    COMMAND integration_pipeline_test
)
