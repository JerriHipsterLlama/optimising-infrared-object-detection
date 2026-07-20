#include "benchmark_runner.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infrared::jetson {
namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << message << '\n';
    }
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) throw std::runtime_error("Dynamic engine shapes require an explicit profile/input adapter.");
        result *= static_cast<std::size_t>(dims.d[index]);
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

}  // namespace

BenchmarkRunner::BenchmarkRunner(BenchmarkOptions options) : options_(std::move(options)) {
    if (options_.iterations <= 0 || options_.warmup < 0) throw std::invalid_argument("warmup/iterations values are invalid");
}

BenchmarkRunner::~BenchmarkRunner() = default;

void BenchmarkRunner::run() {
    std::ifstream engine_file(options_.engine_path, std::ios::binary);
    if (!engine_file) throw std::runtime_error("Unable to open TensorRT engine: " + options_.engine_path);
    engine_file.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(engine_file.tellg());
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> serialized(size);
    engine_file.read(serialized.data(), static_cast<std::streamsize>(serialized.size()));

    Logger logger;
    auto runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) throw std::runtime_error("TensorRT runtime creation failed");
    auto engine = runtime->deserializeCudaEngine(serialized.data(), serialized.size());
    if (!engine) throw std::runtime_error("TensorRT engine deserialization failed");
    auto context = engine->createExecutionContext();
    if (!context) throw std::runtime_error("TensorRT execution-context creation failed");

    std::vector<void*> buffers(engine->getNbIOTensors(), nullptr);
    std::vector<std::size_t> tensor_bytes(engine->getNbIOTensors(), 0);
    std::size_t input_bytes = 0;
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
        const char* name = engine->getIOTensorName(index);
        const auto dims = engine->getTensorShape(name);
        const auto bytes = volume(dims) * sizeof(float);
        tensor_bytes[index] = bytes;
        check_cuda(cudaMalloc(&buffers[index], bytes), "cudaMalloc");
        context->setTensorAddress(name, buffers[index]);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_bytes = bytes;
    }
    if (input_bytes == 0) throw std::runtime_error("TensorRT engine has no input tensor");
    std::vector<char> host_input(input_bytes, 0);
    for (int iteration = 0; iteration < options_.warmup; ++iteration) {
        for (int index = 0; index < engine->getNbIOTensors(); ++index) {
            const char* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                check_cuda(cudaMemcpy(buffers[index], host_input.data(), input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
            }
        }
        if (!context->enqueueV3(nullptr)) throw std::runtime_error("TensorRT warmup enqueue failed");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    std::vector<double> timings;
    std::vector<double> preprocess_timings;
    std::vector<double> inference_timings;
    std::vector<double> postprocess_timings;
    timings.reserve(options_.iterations);
    preprocess_timings.reserve(options_.iterations);
    inference_timings.reserve(options_.iterations);
    postprocess_timings.reserve(options_.iterations);
    for (int iteration = 0; iteration < options_.iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const auto preprocess_start = std::chrono::steady_clock::now();
        for (int index = 0; index < engine->getNbIOTensors(); ++index) {
            const char* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                check_cuda(cudaMemcpy(buffers[index], host_input.data(), input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
            }
        }
        const auto inference_start = std::chrono::steady_clock::now();
        preprocess_timings.push_back(std::chrono::duration<double, std::milli>(inference_start - preprocess_start).count());
        if (!context->enqueueV3(nullptr)) throw std::runtime_error("TensorRT enqueue failed");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        const auto postprocess_start = std::chrono::steady_clock::now();
        for (int index = 0; index < engine->getNbIOTensors(); ++index) {
            const char* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                std::vector<char> host_output(tensor_bytes[index]);
                check_cuda(cudaMemcpy(host_output.data(), buffers[index], tensor_bytes[index], cudaMemcpyDeviceToHost), "cudaMemcpy");
            }
        }
        const auto end = std::chrono::steady_clock::now();
        inference_timings.push_back(std::chrono::duration<double, std::milli>(postprocess_start - inference_start).count());
        postprocess_timings.push_back(std::chrono::duration<double, std::milli>(end - postprocess_start).count());
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    for (void* buffer : buffers) cudaFree(buffer);

    const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    const double p50 = percentile(timings, 0.50);
    const double p95 = percentile(timings, 0.95);
    const double preprocess_mean = std::accumulate(preprocess_timings.begin(), preprocess_timings.end(), 0.0) / preprocess_timings.size();
    const double inference_mean = std::accumulate(inference_timings.begin(), inference_timings.end(), 0.0) / inference_timings.size();
    const double postprocess_mean = std::accumulate(postprocess_timings.begin(), postprocess_timings.end(), 0.0) / postprocess_timings.size();
    std::ofstream output(options_.output_json);
    if (!output) throw std::runtime_error("Unable to write benchmark output: " + options_.output_json);
    output << "{\n"
           << "  \"engine_path\": \"" << options_.engine_path << "\",\n"
           << "  \"batch_size\": 1,\n"
           << "  \"warmup\": " << options_.warmup << ",\n"
           << "  \"sample_count\": " << timings.size() << ",\n"
           << "  \"preprocess_mean_ms\": " << preprocess_mean << ",\n"
           << "  \"inference_mean_ms\": " << inference_mean << ",\n"
           << "  \"postprocess_mean_ms\": " << postprocess_mean << ",\n"
           << "  \"latency_mean_ms\": " << mean << ",\n"
           << "  \"latency_p50_ms\": " << p50 << ",\n"
           << "  \"latency_p95_ms\": " << p95 << ",\n"
           << "  \"fps\": " << (mean > 0.0 ? 1000.0 / mean : 0.0) << "\n"
           << "}\n";
}

}  // namespace infrared::jetson
