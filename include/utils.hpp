/**
 * @brief 边界处理类型
 */
enum class BorderMode {
    ZERO,      // 外围填充值 0 的像素
    CONSTANT,  // 常量填充
    CLAMP,     // 将边界外用最邻近的边界像素值来填充
    REFLECT,   // 将图像或信号像镜子一样反射出去来填充边界
    SYMMETRIC, // 对称填充
    VALID,     // 无填充，仅有效区域
    SAME       // 输出尺寸与输入相同
};
/**
 * @brief 处理卷积设备
 */
enum class CONTYPE{
    CONCPU,             //cpu卷积
    CONGPU_STREAM,      //gpu流处理
    CONGPU_GLOABL_MEM, //gpu全局内存处理
    CONGPU_SHARED_MEM, //gpu共享内存+常量内存
};