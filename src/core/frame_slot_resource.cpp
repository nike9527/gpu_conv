#include "core/frame_slot_resource.hpp"

void frame_slot_resource::init(int w, int h)
{
    width = w;
    height = h;
    size = width * height * sizeof(uchar4);

    // CUDA
    cudaStreamCreate(&stream);
    cudaEventCreate(&done);

    // OpenGL PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // CUDA-GL interop
    cudaGraphicsGLRegisterBuffer(
        &cuda_pbo,
        pbo,
        cudaGraphicsRegisterFlagsWriteDiscard);
}

void frame_slot_resource::acquire_cuda()
{
    if (state != State::Free)
        throw std::runtime_error("Slot not free");

    cudaGraphicsMapResources(1, &cuda_pbo, stream);
    cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void **>(&d_ptr),
        &size,
        cuda_pbo);

    state = State::InFlight;
}

void frame_slot_resource::release_cuda()
{
    cudaGraphicsUnmapResources(1, &cuda_pbo, stream);
    cudaEventRecord(done, stream);
}

bool frame_slot_resource::poll_ready()
{
    if (state != State::InFlight)
        return false;

    if (cudaEventQuery(done) == cudaSuccess)
    {
        state = State::Ready;
        return true;
    }
    return false;
}

void frame_slot_resource::reset()
{
    state = State::Free;
    d_ptr = nullptr;
}

void frame_slot_resource::destroy()
{
    cudaGraphicsUnregisterResource(cuda_pbo);
    glDeleteBuffers(1, &pbo);
    cudaEventDestroy(done);
    cudaStreamDestroy(stream);
}
