#include "benchmark_runner.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infrared::jetson {
namespace {

namespace fs = std::filesystem;

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << message << '\n';
    }
};

struct TensorBuffer {
    std::string name;
    nvinfer1::TensorIOMode mode;
    nvinfer1::Dims dims;
    nvinfer1::DataType type;
    std::size_t bytes{0};
    void* device{nullptr};
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::runtime_error("Dynamic TensorRT shapes are unsupported; export a fixed batch-1 engine.");
        }
        result *= static_cast<std::size_t>(dims.d[index]);
    }
    return result;
}

std::size_t element_size(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(std::uint16_t);
    case nvinfer1::DataType::kINT32: return sizeof(std::int32_t);
    case nvinfer1::DataType::kINT64: return sizeof(std::int64_t);
    case nvinfer1::DataType::kBOOL: return sizeof(bool);
    default: throw std::runtime_error("Unsupported TensorRT tensor data type.");
    }
}

std::string data_type_name(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return "float32";
    case nvinfer1::DataType::kHALF: return "float16";
    case nvinfer1::DataType::kINT32: return "int32";
    case nvinfer1::DataType::kINT64: return "int64";
    case nvinfer1::DataType::kBOOL: return "bool";
    default: return "unknown";
    }
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const int exponent = static_cast<int>((bits >> 23U) & 0xffU) - 127;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= -15) return sign;
    if (exponent >= 16) return static_cast<std::uint16_t>(sign | 0x7c00U);
    return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>((exponent + 15) << 10U) | static_cast<std::uint16_t>(mantissa >> 13U));
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped << '\\';
        if (character == '\n') escaped << "\\n";
        else if (character == '\r') escaped << "\\r";
        else escaped << character;
    }
    return escaped.str();
}

std::vector<fs::path> image_files(const fs::path& directory) {
    if (!fs::is_directory(directory)) {
        throw std::runtime_error("Input directory does not exist: " + directory.string());
    }
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        auto extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("No .jpg, .jpeg, or .png images found in: " + directory.string());
    return files;
}

std::vector<float> preprocess(const fs::path& path, int channels, int height, int width) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("Unable to decode image: " + path.string());
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
    std::vector<float> chw(static_cast<std::size_t>(channels) * height * width);
    for (int channel = 0; channel < channels; ++channel) {
        auto* destination = chw.data() + static_cast<std::size_t>(channel) * height * width;
        for (int row = 0; row < height; ++row) {
            const auto* source = resized.ptr<unsigned char>(row);
            for (int column = 0; column < width; ++column) {
                destination[row * width + column] = static_cast<float>(source[column]) / 255.0F;
            }
        }
    }
    return chw;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

std::string dims_json(const nvinfer1::Dims& dims) {
    std::ostringstream result;
    result << '[';
    for (int index = 0; index < dims.nbDims; ++index) {
        if (index != 0) result << ',';
        result << dims.d[index];
    }
    result << ']';
    return result.str();
}

void release_buffers(std::vector<TensorBuffer>& tensors) {
    for (auto& tensor : tensors) {
        if (tensor.device != nullptr) cudaFree(tensor.device);
        tensor.device = nullptr;
    }
}

}  // namespace

BenchmarkRunner::BenchmarkRunner(BenchmarkOptions options) : options_(std::move(options)) {
    if (options_.iterations <= 0 || options_.warmup < 0) {
        throw std::invalid_argument("warmup must be non-negative and iterations must be positive");
    }
}

BenchmarkRunner::~BenchmarkRunner() = default;

void BenchmarkRunner::run() {
    const auto images = image_files(options_.input_directory);
    const auto output_parent = fs::path(options_.output_json).parent_path();
    if (!output_parent.empty()) fs::create_directories(output_parent);
    if (!options_.raw_output_directory.empty()) fs::create_directories(options_.raw_output_directory);

    std::ifstream engine_file(options_.engine_path, std::ios::binary);
    if (!engine_file) throw std::runtime_error("Unable to open TensorRT engine: " + options_.engine_path);
    engine_file.seekg(0, std::ios::end);
    const auto engine_size = static_cast<std::size_t>(engine_file.tellg());
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> serialized(engine_size);
    engine_file.read(serialized.data(), static_cast<std::streamsize>(serialized.size()));

    Logger logger;
    auto runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) throw std::runtime_error("TensorRT runtime creation failed");
    auto engine = runtime->deserializeCudaEngine(serialized.data(), serialized.size());
    if (!engine) throw std::runtime_error("TensorRT engine deserialization failed");
    auto context = engine->createExecutionContext();
    if (!context) throw std::runtime_error("TensorRT execution-context creation failed");

    std::vector<TensorBuffer> tensors;
    tensors.reserve(engine->getNbIOTensors());
    TensorBuffer* input = nullptr;
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
        TensorBuffer tensor;
        tensor.name = engine->getIOTensorName(index);
        tensor.mode = engine->getTensorIOMode(tensor.name.c_str());
        tensor.dims = engine->getTensorShape(tensor.name.c_str());
        tensor.type = engine->getTensorDataType(tensor.name.c_str());
        tensor.bytes = volume(tensor.dims) * element_size(tensor.type);
        check_cuda(cudaMalloc(&tensor.device, tensor.bytes), "cudaMalloc");
        if (!context->setTensorAddress(tensor.name.c_str(), tensor.device)) {
            release_buffers(tensors);
            cudaFree(tensor.device);
            throw std::runtime_error("TensorRT rejected tensor address: " + tensor.name);
        }
        tensors.push_back(tensor);
        if (tensor.mode == nvinfer1::TensorIOMode::kINPUT) {
            if (input != nullptr) {
                release_buffers(tensors);
                throw std::runtime_error("Only one input tensor is supported by this benchmark.");
            }
            input = &tensors.back();
        }
    }
    if (input == nullptr || input->dims.nbDims != 4) {
        release_buffers(tensors);
        throw std::runtime_error("Expected one static NCHW input tensor.");
    }
    if (input->dims.d[0] != 1 || (input->type != nvinfer1::DataType::kFLOAT && input->type != nvinfer1::DataType::kHALF)) {
        release_buffers(tensors);
        throw std::runtime_error("Expected a batch-1 FP32 or FP16 input tensor.");
    }
    const int channels = input->dims.d[1];
    const int height = input->dims.d[2];
    const int width = input->dims.d[3];
    if (channels != 1 && channels != 3) {
        release_buffers(tensors);
        throw std::runtime_error("Only one- or three-channel image inputs are supported.");
    }

    std::vector<float> host_float_input(static_cast<std::size_t>(channels) * height * width);
    std::vector<std::uint16_t> host_half_input(host_float_input.size());
    std::vector<char> host_output;
    std::vector<double> preprocess_timings;
    std::vector<double> h2d_timings;
    std::vector<double> inference_timings;
    std::vector<double> d2h_timings;
    std::vector<double> postprocess_timings;
    std::vector<double> total_timings;
    preprocess_timings.reserve(options_.iterations);
    h2d_timings.reserve(options_.iterations);
    inference_timings.reserve(options_.iterations);
    d2h_timings.reserve(options_.iterations);
    postprocess_timings.reserve(options_.iterations);
    total_timings.reserve(options_.iterations);

    auto prepare_input = [&](const fs::path& image_path) {
        host_float_input = preprocess(image_path, channels, height, width);
        if (input->type == nvinfer1::DataType::kHALF) {
            for (std::size_t index = 0; index < host_float_input.size(); ++index) {
                host_half_input[index] = float_to_half(host_float_input[index]);
            }
        }
    };

    auto run_once = [&](const fs::path& image_path, int iteration, bool measured) {
        const auto total_start = std::chrono::steady_clock::now();
        const auto preprocess_start = total_start;
        prepare_input(image_path);
        const auto h2d_start = std::chrono::steady_clock::now();
        const void* source = input->type == nvinfer1::DataType::kHALF ? static_cast<const void*>(host_half_input.data()) : static_cast<const void*>(host_float_input.data());
        check_cuda(cudaMemcpy(input->device, source, input->bytes, cudaMemcpyHostToDevice), "cudaMemcpy host-to-device");
        const auto inference_start = std::chrono::steady_clock::now();
        if (!context->enqueueV3(nullptr)) throw std::runtime_error("TensorRT enqueue failed");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        const auto d2h_start = std::chrono::steady_clock::now();
        std::vector<std::vector<char>> output_data;
        for (const auto& tensor : tensors) {
            if (tensor.mode != nvinfer1::TensorIOMode::kOUTPUT) continue;
            output_data.emplace_back(tensor.bytes);
            check_cuda(cudaMemcpy(output_data.back().data(), tensor.device, tensor.bytes, cudaMemcpyDeviceToHost), "cudaMemcpy device-to-host");
        }
        const auto postprocess_start = std::chrono::steady_clock::now();
        if (measured && !options_.raw_output_directory.empty()) {
            std::ostringstream stem;
            stem << std::setw(06) << std::setfill('0') << iteration;
            for (std::size_t output_index = 0; output_index < output_data.size(); ++output_index) {
                std::ofstream raw(fs::path(options_.raw_output_directory) / (stem.str() + "_output_" + std::to_string(output_index) + ".bin"), std::ios::binary);
                if (!raw) throw std::runtime_error("Unable to write raw TensorRT output.");
                raw.write(output_data[output_index].data(), static_cast<std::streamsize>(output_data[output_index].size()));
            }
        }
        const auto end = std::chrono::steady_clock::now();
        if (measured) {
            preprocess_timings.push_back(std::chrono::duration<double, std::milli>(h2d_start - preprocess_start).count());
            h2d_timings.push_back(std::chrono::duration<double, std::milli>(inference_start - h2d_start).count());
            inference_timings.push_back(std::chrono::duration<double, std::milli>(d2h_start - inference_start).count());
            d2h_timings.push_back(std::chrono::duration<double, std::milli>(postprocess_start - d2h_start).count());
            postprocess_timings.push_back(std::chrono::duration<double, std::milli>(end - postprocess_start).count());
            total_timings.push_back(std::chrono::duration<double, std::milli>(end - total_start).count());
        }
    };

    for (int iteration = 0; iteration < options_.warmup; ++iteration) run_once(images[static_cast<std::size_t>(iteration) % images.size()], iteration, false);
    for (int iteration = 0; iteration < options_.iterations; ++iteration) run_once(images[static_cast<std::size_t>(iteration) % images.size()], iteration, true);

    release_buffers(tensors);
    fs::path metadata_path;
    if (!options_.raw_output_directory.empty()) {
        metadata_path = fs::path(options_.raw_output_directory) / "metadata.json";
        std::ofstream metadata(metadata_path);
        metadata << "{\n  \"engine_path\": \"" << json_escape(options_.engine_path) << "\",\n  \"images\": [\n";
        for (std::size_t index = 0; index < images.size(); ++index) {
            metadata << "    {\"path\": \"" << json_escape(images[index].string()) << "\"}" << (index + 1 == images.size() ? "\n" : ",\n");
        }
        metadata << "  ],\n  \"tensors\": [\n";
        bool first_output = true;
        for (const auto& tensor : tensors) {
            if (tensor.mode != nvinfer1::TensorIOMode::kOUTPUT) continue;
            if (!first_output) metadata << ",\n";
            metadata << "    {\"name\": \"" << json_escape(tensor.name) << "\", \"mode\": \"output\", \"shape\": " << dims_json(tensor.dims) << ", \"dtype\": \"" << data_type_name(tensor.type) << "\"}";
            first_output = false;
        }
        metadata << "\n  ],\n  \"samples\": [\n";
        int output_count = 0;
        for (const auto& tensor : tensors) {
            if (tensor.mode == nvinfer1::TensorIOMode::kOUTPUT) ++output_count;
        }
        for (int iteration = 0; iteration < options_.iterations; ++iteration) {
            std::ostringstream stem;
            stem << std::setw(06) << std::setfill('0') << iteration;
            metadata << "    {\"iteration\": " << iteration
                    << ", \"image\": \"" << json_escape(images[static_cast<std::size_t>(iteration) % images.size()].string())
                    << "\", \"outputs\": [";
            for (int output_index = 0; output_index < output_count; ++output_index) {
                if (output_index != 0) metadata << ", ";
                metadata << "\"" << stem.str() << "_output_" << output_index << ".bin\"";
            }
            metadata << "]}" << (iteration + 1 == options_.iterations ? "\n" : ",\n");
        }
        metadata << "  ]\n}\n";
    }

    std::ofstream output(options_.output_json);
    if (!output) throw std::runtime_error("Unable to write benchmark output: " + options_.output_json);
    const double total_mean = mean(total_timings);
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"engine_path\": \"" << json_escape(options_.engine_path) << "\",\n"
           << "  \"batch_size\": 1,\n"
           << "  \"warmup\": " << options_.warmup << ",\n"
           << "  \"image_count\": " << images.size() << ",\n"
           << "  \"sample_count\": " << total_timings.size() << ",\n"
           << "  \"preprocess_mean_ms\": " << mean(preprocess_timings) << ",\n"
           << "  \"h2d_mean_ms\": " << mean(h2d_timings) << ",\n"
           << "  \"inference_mean_ms\": " << mean(inference_timings) << ",\n"
           << "  \"d2h_mean_ms\": " << mean(d2h_timings) << ",\n"
           << "  \"postprocess_mean_ms\": " << mean(postprocess_timings) << ",\n"
           << "  \"latency_mean_ms\": " << total_mean << ",\n"
           << "  \"latency_p50_ms\": " << percentile(total_timings, 0.50) << ",\n"
           << "  \"latency_p95_ms\": " << percentile(total_timings, 0.95) << ",\n"
           << "  \"fps\": " << (total_mean > 0.0 ? 1000.0 / total_mean : 0.0);
    if (!metadata_path.empty()) output << ",\n  \"raw_output_metadata\": \"" << json_escape(metadata_path.string()) << "\"";
    output << "\n}\n";
}

}  // namespace infrared::jetson
