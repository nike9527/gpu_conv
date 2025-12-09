void conv2d_cpu_omp(const float* in, float* out, const int w, const int h, const float* kernel, const int ksize);
/**
 * @brief 高斯卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param sigma 
 * @param kSize 积核大小
 */
void gaussianConvolution(const float* in, float* out, const int w, const int h, const int kSize, const float sigma);

/**
 * @brief  sobel 卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param dx x 方向卷积
 * @param dy y 方向卷积
 */
void sobelConvolution(const float* in, float* out, const int w, const int h,const int dx, const int dy);

/**
 * @brief  sobel 卷积
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param dx x 方向卷积
 * @param dy y 方向卷积
 */
void sobelConvolution(const float* in, float* out, const int w, const int h,const int dx, const int dy);
/**
 * @brief  锐化滤波器
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 */
void sharpenConvolution(const float* in, float* out, const int w, const int h);

/**
 * @brief 均值模糊
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 * @param ksize 内核大小
 */
void meanBlurConvolution(const float* in, float* out, const int w, const int h,int const ksize);

/**
 * @brief 拉普拉斯算子
 * @param in  输入数据
 * @param out 输入数据
 * @param w   高度  
 * @param h   宽度
 */
void laplacianConvolution(const float* in, float* out, const int w, const int h);