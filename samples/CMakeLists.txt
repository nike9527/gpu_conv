add_executable(samples_cpu_vs_gpu src/cpu_vs_gpu_benchmark.cpp src/filter.cpp)
target_include_directories(samples_cpu_vs_gpu
    PRIVATE  
    ${CMAKE_SOURCE_DIR}/include                          # gconv 的使用者也会包含
    ${CMAKE_SOURCE_DIR}/utils/image_viewer/include       # 只对 gconv 内部可见
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(samples_cpu_vs_gpu PRIVATE gconv image_viewer)
