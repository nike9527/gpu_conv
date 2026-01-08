#pragma once
enum class mem_type
{
    GLOBAL,
    SHAREDCONST
};
enum class dev_type
{
    DEVCPU,
    DEVGPU
};
enum class filter_type
{
    GAUSSIAN,
    SOBEL,
    SOBELX,
    SOBELY,
    SHARPEN,
    MEANBLUR,
    LAPLACIAN,
    FILTERCUSTOM
};
struct kernel_desc
{
    dev_type dev_type;
    filter_type filter;
    mem_type mem_type;
    int ksize;
};
