#include <array>
#include "cuda_memory.hpp"
struct frame_buffer {
    cuda_memory<float> input;
    cuda_memory<float> output;

    frame_buffer(size_t inSize, size_t outSize): input(inSize), output(outSize) {}
};

class triple_pipeline {
public:
    triple_pipeline(size_t inSize, size_t outSize)
        : frames{frame_buffer(inSize, outSize),frame_buffer(inSize, outSize),frame_buffer(inSize, outSize)} 
    {}

    frame_buffer& next() {
        idx = (idx + 1) % 3;
        return frames[idx];
    }

private:
    std::array<frame_buffer, 3> frames;
    int idx = 0;
};
