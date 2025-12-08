/**
 * @brief 高斯卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param sigma 
 * @param kSize 积核大小
 */
extern void gaussianConvolutionGPU(const float* in, float* out, const int w, const int h, const int kSize, const float sigma);
/**
 * @brief sobel卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param dx x方向卷积 0不做处理
 * @param dy y方向卷积 0不做处理
 * */
extern void sobelConvolutionGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy);