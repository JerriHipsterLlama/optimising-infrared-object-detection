#include "benchmark_runner.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* executable) {
    std::cout << "Usage: " << executable
              << " --engine PATH --input-dir PATH --output-json PATH"
                 " [--raw-output-dir PATH] [--warmup N] [--iterations N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    infrared::jetson::BenchmarkOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            return argv[++index];
        };
        if (argument == "--help") {
            usage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (argument == "--engine") options.engine_path = value();
        else if (argument == "--input-dir") options.input_directory = value();
        else if (argument == "--output-json") options.output_json = value();
        else if (argument == "--raw-output-dir") options.raw_output_directory = value();
        else if (argument == "--warmup") options.warmup = std::stoi(value());
        else if (argument == "--iterations") options.iterations = std::stoi(value());
        else throw std::invalid_argument("unknown argument: " + argument);
    }
    if (options.engine_path.empty() || options.output_json.empty() || options.input_directory.empty()) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    try {
        infrared::jetson::BenchmarkRunner runner(std::move(options));
        runner.run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
