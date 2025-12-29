#pragma once
#include <cmath>
#include <gtest/gtest.h>

inline bool nearly_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

inline void expect_image_near(const float* ref,const float* out,int size,float eps = 1e-4f) {
    for (int i = 0; i < size; ++i) {
        ASSERT_TRUE(nearly_equal(ref[i], out[i], eps))<< "Mismatch at " << i<< " ref=" << ref[i]<< " out=" << out[i];
    }
}
