#pragma once
#include <vector>

void cpu_convolution(const float* in,float* out,int width,int height,const std::vector<float>& kernel,int ksize);
