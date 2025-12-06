void conv2d_cpu_omp(const float* in, float* out, int w, int h, const float* kernel, int ksize);
/**
 * @brief 高斯卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param kernel 卷积内核
 * @param kSize 积核大小
 */
void gaussianConvolution(const float* in, float* out, int w, int h, const float* kernel, int kSize);