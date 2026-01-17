cmake_minimum_required(VERSION 3.10)
project(glad)

# 创建GLAD库
add_library(glad
    src/glad.c
)

# 包含目录
target_include_directories(glad
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接OpenGL库
if(WIN32)
    target_link_libraries(glad opengl32)
elseif(APPLE)
    target_link_libraries(glad "-framework OpenGL")
else()
    target_link_libraries(glad GL)
endif()