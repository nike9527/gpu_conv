#pragma once
#include <vector>
struct image_info
{
    int width = 0;
    int height = 0;
    int channels = 1;
    std::vector<float> data;
    image_info(int w, int h, int c) : width(w), height(h), channels(c), data(0.f, w * h * c)
    {
    }
};