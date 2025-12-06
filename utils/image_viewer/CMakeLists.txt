cmake_minimum_required(VERSION 3.18)

#-----------------------------------
# 包含路径
#-----------------------------------
set(SDL3_DIR       "${CMAKE_SOURCE_DIR}/external/SDL3")
set(SDL3_IMAGE_DIR "${CMAKE_SOURCE_DIR}/external/SDL3_image")
#-----------------------------------
# 1. image_viewer 静态库
#-----------------------------------
add_library(image_viewer STATIC 
    src/image_viewer.cpp
)
target_include_directories(image_viewer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
#-----------------------------------
# 2. 链接 SDL3 和 SDL3_image
#-----------------------------------
if(TARGET SDL3::SDL3)
    target_link_libraries(image_viewer PUBLIC SDL3::SDL3)
elseif(TARGET SDL3)
    target_link_libraries(image_viewer PUBLIC SDL3)
endif()

if(TARGET SDL3_image::SDL3_image)
    target_link_libraries(image_viewer PUBLIC SDL3_image::SDL3_image)
endif()