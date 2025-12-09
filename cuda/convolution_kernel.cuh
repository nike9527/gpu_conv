
/**
 * @brief 自定义卷积
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 * @return __global__ 
 */
 __global__ void conv2d_global_kernel(const float* input, float* output, const int width, const int height, const float * kernel, const int kSize);

 /**
  * @brief 高斯模糊
  * @param input 输入数据
  * @param output 输出数据
  * @param width 宽度
  * @param height 高度
  * @param kernel 内核
  * @param ksize 内核大小
  * @return __global__ 
  */
__global__ void gaussianConvolution(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * const kernel, const int kSize);
/**
 * @brief sobel算子
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 * @param kernelX 内核 X方向
 * @param kernelY 内核 Y方向
 * @param kSize 内核大小
 * @return __global__ 
 */
__global__ void sobelConvolution(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * const kernelX, const float * const kernelY, const int kSize);
/**
 * @brief 锐化滤波器
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void sharpenConvolution(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * kernel, const int kSize);
/**
 * @brief 均值模糊
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void meanBlurConvolution(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * kernel, const int kSize);
/**
 * @brief 拉普拉斯算子
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void laplacianConvolution(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * kernel, const int kSize);

/**
 * @brief 自定义卷积
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 * @return __global__ 
 */
__global__ void conv2d_global_kernelWithShared(const float* input, float* output, const int width, const int height, const float * kernel, const int kSize);
 /**
  * @brief 高斯模糊
  * @param input 输入数据
  * @param output 输出数据
  * @param width 宽度
  * @param height 高度
  * @param kernel 内核
  * @param ksize 内核大小
  * @return __global__ 
  */
__global__ void gaussianConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * const kernel, const int kSize);
/**
 * @brief sobel算子
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 * @param kernelX 内核 X方向
 * @param kernelY 内核 Y方向
 * @param kSize 内核大小
 * @return __global__ 
 */
__global__ void sobelConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * const kernelX, const float * const kernelY, const int kSize);
/**
 * @brief 锐化滤波器
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void sharpenConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * kernel, const int kSize);
/**
 * @brief 均值模糊
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void meanBlurConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * kernel, const int kSize);
/**
 * @brief 拉普拉斯算子
 * @param input 输入数据
 * @param output 输出数据
 * @param width 宽度
 * @param height 高度
 * @param kernel 内核
 * @param ksize 内核大小
 */
__global__ void laplacianConvolutionWithShared(const float* __restrict__ input, float* __restrict__ output, const int width, const int height, const float * kernel, const int kSize);