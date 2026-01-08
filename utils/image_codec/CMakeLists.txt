cmake_minimum_required(VERSION 3.18)
project(image_codec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-----------------------------------
# 包含路径
#-----------------------------------
set(STB_IMAGE "${CMAKE_SOURCE_DIR}/external/stb_image")
#-----------------------------------
# 1. image_codec 静态库
#-----------------------------------
add_library(image_codec STATIC src/image_codec.cpp)
target_include_directories(image_codec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${STB_IMAGE})