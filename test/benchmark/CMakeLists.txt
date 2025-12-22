add_executable(bench_test bench_main.cpp bench_types.hpp bench_timer.hpp bench_runner.cu bench_report.cpp bench_config.hpp bench_filter.hpp)
target_include_directories(bench_test
    PRIVATE  
    ${CMAKE_SOURCE_DIR}/include                         
    ${CMAKE_SOURCE_DIR}/utils/image_viewer/include       
    ${CMAKE_SOURCE_DIR}/cuda       
)

target_link_libraries(bench_test PRIVATE gconv)
