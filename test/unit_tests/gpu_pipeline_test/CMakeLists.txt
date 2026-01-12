add_executable(test_triple_pipeline
    test_triple_pipeline.cpp
    test_kernel.cu
)

set_target_properties(test_triple_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(test_triple_pipeline
    PRIVATE
    gconv
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)
copy_runtime_dlls(test_triple_pipeline)
add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
include(GoogleTest)
if (WIN32)
gtest_discover_tests(test_triple_pipeline
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    DISCOVERY_MODE POST_BUILD
)
else()
    gtest_discover_tests(test_triple_pipeline)
endif()