#===========================gpu 全局内存===================================
add_executable(unit_test_gpu_kernel_global
    unit_test_gpu_kernel_global.cpp
)

set_target_properties(unit_test_gpu_kernel_global PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(unit_test_gpu_kernel_global
    PRIVATE
    gconv
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)
copy_runtime_dlls(unit_test_gpu_kernel_global)
include(GoogleTest)
if (WIN32)
gtest_discover_tests(unit_test_gpu_kernel_global
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    DISCOVERY_MODE PRE_TEST
)
else()
    gtest_discover_tests(unit_test_gpu_kernel_global)
endif()
#===========================gpu 共享内存===================================
add_executable(unit_test_gpu_kernel_shared
    unit_test_gpu_kernel_shared.cpp
)

set_target_properties(unit_test_gpu_kernel_shared PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(unit_test_gpu_kernel_shared
    PRIVATE
    gconv
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)
copy_runtime_dlls(unit_test_gpu_kernel_shared)
include(GoogleTest)
if (WIN32)
gtest_discover_tests(unit_test_gpu_kernel_shared
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    DISCOVERY_MODE POST_BUILD
)
else()
    gtest_discover_tests(unit_test_gpu_kernel_shared)
endif()