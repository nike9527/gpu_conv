#include "conv_dispatcher.hpp"
#include "kernels/kernel_gaussian.hpp"
#include "kernels/kernel_sharpen.hpp"
#include "kernels/kernel_sobel.hpp"
kernel_base* conv_dispatcher::get(const kernel_desc& desc) {
    auto it = cache.find(desc);
    if (it != cache.end())
        it->second.get()->launch(desc);;

    std::unique_ptr<kernel_base> k;

    if (desc.filter == filter_type::SOBEL && desc.mem == mem_type::GLOBAL)
        k = std::make_unique<kernel_sobel>();

    else if (desc.filter == filter_type::SHARPEN && desc.mem == mem_type::SHAREDCONST)
        k = std::make_unique<kernel_sharpen>();

    else if (desc.filter == filter_type::GAUSSIAN && desc.mem == mem_type::GLOBAL)
        k = std::make_unique<kernel_gaussian>(5.0, desc.ksize);

    auto ptr = k.get();
    cache.emplace(desc, std::move(k));
    return ptr;
}
