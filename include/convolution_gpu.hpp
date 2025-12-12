//=============================全局内存=======================================
/**
 * @brief 自定义卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param kSize 核大小
 * @param kernel 核
 */
extern void conv2dGlobalGPU(const float* in, float* out, const int w, const int h, const int kSize, const float* kernel);
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
/**
 * @brief 锐化滤波器
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 */
extern void sharpenConvolutionGPU(const float* in, float* out, const int w, const int h);
/**
 * @brief 均值模糊
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param ksize 内核大小
 */
extern void meanBlurConvolutionGPU(const float* in, float* out, const int w, const int h,int const kSize);

/**
 * @brief 拉普拉斯算子
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 */
extern void laplacianConvolutionGPU(const float* in, float* out, const int w, const int h);
//=============================共享内存+常量内存=======================================
/**
 * @brief 自定义卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param kSize 核大小
 * @param kernel 核
 */
extern void conv2dWithSharedGPU(const float* in, float* out, const int w, const int h, const int kSize, const float* kernel);
/**
 * @brief 高斯卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param sigma 
 * @param kSize 积核大小
 */
extern void gaussianConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h, const int kSize, const float sigma);
/**
 * @brief sobel卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param dx x方向卷积 0不做处理
 * @param dy y方向卷积 0不做处理
 * */
extern void sobelConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,const int dx, const int dy);
/**
 * @brief 锐化滤波器
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 */
extern void sharpenConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h);
/**
 * @brief 均值模糊
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param ksize 内核大小
 */
extern void meanBlurConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h,int const kSize);

/**
 * @brief 拉普拉斯算子
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 */
extern void laplacianConvolutionWithSharedGPU(const float* in, float* out, const int w, const int h);