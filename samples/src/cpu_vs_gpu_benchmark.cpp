/**
 * CPU baseline vs GPU 对比
 */

#include <iostream>
#include "filter.hpp"
#include <vector>
#include <chrono>
#include <string>

int main(int argc, char* argv[]) {
    gconv::gaussianFilter();
    return 0;
}
