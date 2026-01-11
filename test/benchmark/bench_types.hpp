#pragma once
#include <string>
#include "filters/filter.hpp"

enum class PipelineType
{
    SINGLE_STREAM,
    TRIPLE_BUFFER
};

inline const char *toString(mem_type k)
{
    return k == mem_type::GLOBAL ? "Global" : "Shared_Const";
}

inline const filter getFilter(filter_type f, int ksize = 3)
{
    switch (f)
    {
    case filter_type::SOBELX:
        return filter::sobelX();
    case filter_type::SOBELY:
        return filter::sobelY();
    case filter_type::GAUSSIAN:
        return filter::gaussian2D(ksize, 3.0);
    case filter_type::MEANBLUR:
        return filter::meanBlur(ksize);
    case filter_type::SHARPEN:
        return filter::sharpen();
    case filter_type::LAPLACIAN:
        return filter::laplacian();
    default:
        return filter::filterCustom(3, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    }
}

inline const char *toString(PipelineType p)
{
    return p == PipelineType::SINGLE_STREAM ? "Single" : "Triple";
}
