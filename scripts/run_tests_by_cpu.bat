@echo off
echo === 卷积测试套件运行器 ===
echo.

set TEST_EXE=D:\C++\gpu_conv\bin\Debug\unit_cpu_tests.exe

:menu
echo.
echo 请选择要运行的测试类型：
echo 1. 高斯核性能测试
echo 2. 均值模糊性能测试
echo 3. 锐化核测试
echo 4. 退出
echo.

set /p choice="请输入选择 (1-8): "


if "%choice%"=="1" (
    echo 运行高斯核测试...
    "%TEST_EXE%" --gtest_filter="*GaussianTests*" --gtest_print_time=1
    goto menu
)

if "%choice%"=="2" (
    echo 运行均值模糊测试...
    "%TEST_EXE%" --gtest_filter="*MeanBlurTests*" --gtest_print_time=1
    goto menu
)

if "%choice%"=="3" (
    echo 运行锐化核测试...
    "%TEST_EXE%" --gtest_filter="*SharpenTests*" --gtest_print_time=1
    goto menu
)
if "%choice%"=="4" (
    echo 退出
    exit /b 0
)

echo 无效选择，请重新输入
goto menu