# add_executable(bench_sobel
#     bench_sobel.cu
# )

# target_link_libraries(bench_sobel
#     PRIVATE
#     cuda_conv_lib
#     GTest::gtest
# )

# #---------------------------------------------
# # 不注册 add_test
# #---------------------------------------------
# set_target_properties(bench_sobel
#     PROPERTIES
#     EXCLUDE_FROM_ALL TRUE
# )

add_executable(conv_benchmark
    bench_conv2d.cu
)

target_link_libraries(conv_benchmark
    PRIVATE
    gconv
    CUDA::cudart
)

# 不 add_test

