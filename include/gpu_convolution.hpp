/**
 * cuda函数声明
 */
extern bool gaussianConvolutionGPU(const float* in, float* out, int w, int h, const float* kernel, int ksize);
