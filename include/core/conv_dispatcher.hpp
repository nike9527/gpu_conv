#pragma once
#include <map>
#include <memory>
#include "kernel_base.hpp"
#include "filters/kernel_desc.hpp"

class conv_dispatcher {
public:
    kernel_base* get(const kernel_desc& desc);
private:
    std::map<kernel_desc, std::unique_ptr<kernel_base>> cache;
};
