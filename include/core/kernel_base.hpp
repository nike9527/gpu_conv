#pragma once
#include <omp.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda/cuda_stream.hpp"
#include "filters/filter.hpp"
class kernel_base {
public:
    virtual ~kernel_base() = default;
    virtual void launch(const float* in,float* out,int w, int h, cuda_stream stream, const kernel_desc& desc,const filter& filter) = 0;
};
