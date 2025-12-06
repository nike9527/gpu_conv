
### 3. build.sh (Linux构建脚本)

```bash
#!/bin/bash

# GPU卷积库构建脚本 - Linux版本

set -e  # 出错时退出

echo "========================================"
echo "GPU Convolution Library Build Script"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认构建类型
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_DIR="${HOME}/.local"
BUILD_TESTS="ON"
BUILD_EXAMPLES="ON"
USE_CUDA="ON"
USE_OPENCL="ON"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            echo -e "${YELLOW}Cleaning build directory...${NC}"
            rm -rf ${BUILD_DIR}
            shift
            ;;
        --install-dir=*)
            INSTALL_DIR="${1#*=}"
            shift
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --cuda-only)
            USE_OPENCL="OFF"
            shift
            ;;
        --opencl-only)
            USE_CUDA="OFF"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug           Build in debug mode"
            echo "  --clean           Clean build directory"
            echo "  --install-dir=PATH Set installation directory (default: ~/.local)"
            echo "  --no-tests        Don't build tests"
            echo "  --no-examples     Don't build examples"
            echo "  --cuda-only       Only build CUDA backend"
            echo "  --opencl-only     Only build OpenCL backend"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# 检查必要工具
echo -e "${GREEN}Checking dependencies...${NC}"

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

check_command cmake
check_command make

# 检查CUDA
if [ "$USE_CUDA" = "ON" ]; then
    if ! command -v nvcc &> /dev/null; then
        echo -e "${YELLOW}Warning: nvcc not found, CUDA backend will be disabled${NC}"
        USE_CUDA="OFF"
    fi
fi

# 检查OpenCL
if [ "$USE_OPENCL" = "ON" ]; then
    if [ ! -f "/usr/include/CL/cl.h" ] && [ ! -f "/usr/local/cuda/include/CL/cl.h" ]; then
        echo -e "${YELLOW}Warning: OpenCL headers not found, OpenCL backend will be disabled${NC}"
        USE_OPENCL="OFF"
    fi
fi

# 创建构建目录
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# 生成构建配置
echo -e "${GREEN}Configuring CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DBUILD_TESTS=${BUILD_TESTS} \
    -DBUILD_EXAMPLES=${BUILD_EXAMPLES} \
    -DUSE_CUDA=${USE_CUDA} \
    -DUSE_OPENCL=${USE_OPENCL} \
    -DCMAKE_CXX_COMPILER=g++-9 2>&1 | tee cmake.log

# 编译
echo -e "${GREEN}Building project...${NC}"
make -j$(nproc) 2>&1 | tee build.log

# 运行测试
if [ "$BUILD_TESTS" = "ON" ]; then
    echo -e "${GREEN}Running tests...${NC}"
    ctest --output-on-failure 2>&1 | tee test.log
fi

# 安装
echo -e "${GREEN}Installing to ${INSTALL_DIR}...${NC}"
make install

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Library installed to: ${INSTALL_DIR}"
echo "To use the library in your projects:"
echo "1. Add to your CMakeLists.txt:"
echo "   find_package(gpu_convolution REQUIRED)"
echo "   target_link_libraries(your_target gpu_convolution)"
echo "2. Or compile with:"
echo "   g++ your_program.cpp -lgpu_convolution -I${INSTALL_DIR}/include -L${INSTALL_DIR}/lib"
echo ""

cd ..