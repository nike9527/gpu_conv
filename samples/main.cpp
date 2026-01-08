/**
 * CPU baseline vs GPU 对比
 */

#include <iostream>
#include "action.hpp"
#include <vector>
#include <chrono>
#include <string>

int main(int argc, char* argv[]) {
    gconv::convolve();


// FrameProcessor processor(W, H, Filters::SobelX());

// for (;;) {
//     float input[W*H];
//     float output[W*H];

//     captureFrame(input);

//     processor.submit(input);

//     if (processor.fetch(output)) {
//         render(output);
//     }
// }

// // 退出前
// processor.flush();


    return 0;
}
