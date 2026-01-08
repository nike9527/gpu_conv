add_executable(samples_cpu_vs_gpu main.cpp src/action.cpp)
target_include_directories(samples_cpu_vs_gpu
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
)
# 检查库是否存在
if(TARGET image_viewer)
    message(STATUS "Found target: image_viewer")
else()
    message(WARNING "Target image_viewer not found!")
endif()
# 检查库是否存在
if(TARGET image_codec)
    message(STATUS "Found target: image_codec")
else()
    message(WARNING "Target image_codec not found!")
endif()
target_link_libraries(samples_cpu_vs_gpu PRIVATE gconv image_viewer image_codec)
