#include "core/frame_slot_base.hpp"
#include "cuda/cuda_memory.hpp"
#include "cuda/cuda_memory.hpp"
#include "cuda/cuda_event.hpp"
#include "cuda/cuda_stream.hpp"
class cuda_frame_slot : public frame_slot_base
{
public:
    cuda_frame_slot(int width, int height);
    // 不可拷贝
    cuda_frame_slot(const cuda_frame_slot &) = delete;
    cuda_frame_slot &operator=(const cuda_frame_slot &) = delete;

    int width_ = 0;
    int height_ = 0;
    int frame_id = -1;
    cuda_memory<float> d_in_;
    cuda_memory<float> d_out_;
    cuda_event done_;
    cuda_stream stream_{cudaStreamNonBlocking};
    virtual void mark_submit() override;
    virtual bool is_ready() const override;
    virtual ~cuda_frame_slot();
};