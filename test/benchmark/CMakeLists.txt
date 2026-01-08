add_executable(bench_test bench_main.cpp bench_types.hpp bench_runner.cu bench_report.cpp bench_config.hpp bench_filter.hpp)
target_include_directories(bench_test
    PRIVATE  
    ${CMAKE_SOURCE_DIR}/include                           
    ${CMAKE_SOURCE_DIR}/cuda       
)
# set_target_properties(bench_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(bench_test PRIVATE gconv)

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

# add_executable(conv_benchmark
#     bench_conv2d.cu
# )

# target_link_libraries(conv_benchmark
#     PRIVATE
#     gconv
#     CUDA::cudart
# )

# 不 add_test

