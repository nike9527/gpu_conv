@echo off
REM GPU卷积库构建脚本 - Windows版本

echo ========================================
echo GPU Convolution Library Build Script
echo ========================================

setlocal enabledelayedexpansion

REM 默认值
set BUILD_TYPE=Release
set BUILD_DIR=build
set INSTALL_DIR=%USERPROFILE%\gpu_convolution
set BUILD_TESTS=ON
set BUILD_EXAMPLES=ON
set USE_CUDA=ON
set GENERATOR="Visual Studio 16 2019"

REM 解析参数
:parse_args
if "%1"=="" goto :args_done
if "%1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto :parse_args
)
if "%1"=="--clean" (
    echo Cleaning build directory...
    rmdir /s /q %BUILD_DIR%
    shift
    goto :parse_args
)
if "%1"=="--install-dir" (
    set INSTALL_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--vs2017" (
    set GENERATOR="Visual Studio 15 2017"
    shift
    goto :parse_args
)
if "%1"=="--vs2019" (
    set GENERATOR="Visual Studio 16 2019"
    shift
    goto :parse_args
)
if "%1"=="--vs2022" (
    set GENERATOR="Visual Studio 17 2022"
    shift
    goto :parse_args
)
if "%1"=="--help" (
    echo Usage: %0 [options]
    echo Options:
    echo   --debug           Build in debug mode
    echo   --clean           Clean build directory
    echo   --install-dir DIR Set installation directory
    echo   --vs2017          Use Visual Studio 2017
    echo   --vs2019          Use Visual Studio 2019 ^(default^)
    echo   --vs2022          Use Visual Studio 2022
    echo   --help            Show this help message
    exit /b 0
)
echo Unknown option: %1
exit /b 1
:args_done

REM 检查CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: CMake is not installed or not in PATH
    exit /b 1
)

REM 检查CUDA
if "%USE_CUDA%"=="ON" (
    where nvcc >nul 2>nul
    if %errorlevel% neq 0 (
        echo Warning: nvcc not found, CUDA backend will be disabled
        set USE_CUDA=OFF
    )
)

REM 创建构建目录
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

REM 生成构建配置
echo Configuring CMake...
cmake .. ^
    -G %GENERATOR% ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
    -DBUILD_TESTS=%BUILD_TESTS% ^
    -DBUILD_EXAMPLES=%BUILD_EXAMPLES% ^
    -DUSE_CUDA=%USE_CUDA% ^
    -DUSE_OPENCL=%USE_OPENCL% > cmake.log 2>&1

if %errorlevel% neq 0 (
    echo CMake configuration failed
    type cmake.log
    exit /b 1
)

REM 编译
echo Building project...
cmake --build . --config %BUILD_TYPE% --target ALL_BUILD > build.log 2>&1

if %errorlevel% neq 0 (
    echo Build failed
    type build.log
    exit /b 1
)

REM 运行测试
if "%BUILD_TESTS%"=="ON" (
    echo Running tests...
    ctest -C %BUILD_TYPE% --output-on-failure > test.log 2>&1
)

REM 安装
echo Installing to %INSTALL_DIR%...
cmake --build . --config %BUILD_TYPE% --target INSTALL > install.log 2>&1

echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Library installed to: %INSTALL_DIR%
echo.
echo To use the library in Visual Studio:
echo 1. Add ^"%INSTALL_DIR%\include^" to Additional Include Directories
echo 2. Add ^"%INSTALL_DIR%\lib^" to Additional Library Directories
echo 3. Add ^"gpu_convolution.lib^" to Additional Dependencies
echo.

cd ..
endlocal