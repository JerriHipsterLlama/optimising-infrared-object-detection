#pragma once

#include <cstddef>
#include <string>

namespace infrared::jetson {

struct BenchmarkOptions {
    std::string engine_path;
    std::string input_directory;
    std::string output_json;
    int warmup{20};
    int iterations{100};
};

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkOptions options);
    ~BenchmarkRunner();

    BenchmarkRunner(const BenchmarkRunner&) = delete;
    BenchmarkRunner& operator=(const BenchmarkRunner&) = delete;

    void run();

private:
    BenchmarkOptions options_;
};

}  // namespace infrared::jetson
