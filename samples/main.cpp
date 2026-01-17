/**
 * CPU baseline vs GPU 对比
 */

#include <iostream>
#include "action.hpp"
#include <vector>
#include <chrono>
#include <string>
#include "gl/gl_pbo.hpp"
#include "cuda/cuda_stream.hpp"
#include "GLFW/glfw3.h"
#include <thread>
int main(int argc, char *argv[])
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(800, 800, "Day2 - Fullscreen Quad", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    gconv::gaussianAction();
    return 0;

    // cuda_stream stream;
    // GLPBO pbo(800, 800);
    // pbo.map(stream.get());
    // cudaMemsetAsync(pbo.device_ptr(), 255, pbo.size_bytes(), stream.get());
    // pbo.unmap(stream);
    // while (true)
    // {
    //     std::this_thread::sleep_for(std::chrono::seconds(2));
    // }
}
