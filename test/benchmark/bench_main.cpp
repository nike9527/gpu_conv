#include "bench_config.hpp"
#include "bench_types.hpp"
#include <benchmark/benchmark.h>

extern BenchResult runBenchmark(const BenchCase &);

extern void printResult(const BenchCase &, float, float);

// int main()
// {
//     auto cases = getBenchCases();

//     for (auto &c : cases)
//     {
//         auto r = runBenchmark(c);
//         printResult(c, r.kernel_ms, r.gpixel);
//     }
//     return 0;
// }
int main(int argc, char **argv)
{
    // 设置输出格式
    ::benchmark::ConsoleReporter reporter;

    ::benchmark::RunSpecifiedBenchmarks(&reporter);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}