// #define GLFW_INCLUDE_NONE // 防止GLFW自动包含OpenGL头文件
#pragma once
#if defined(USE_GLES)
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#else
#include <glad/glad.h> // 或 gl.h + glext.h
#endif