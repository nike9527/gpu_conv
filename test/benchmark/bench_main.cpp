#include "bench_config.hpp"
#include "bench_types.hpp"

extern BenchResult runBenchmark(const BenchCase&);

extern void printResult(const BenchCase&, float, float);

int main() {
    auto cases = getBenchCases();

    for (auto& c : cases) {
        auto r = runBenchmark(c);
        printResult(c, r.kernel_ms, r.gpixel);
    }
    return 0;
}
