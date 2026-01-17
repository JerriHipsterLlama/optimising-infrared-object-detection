#include "data_loader.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

CAMELDataLoader::CAMELDataLoader(const std::string& data_path, int batch_size)
    : data_path_(data_path), batch_size_(batch_size), current_index_(0) {
    fs::path images_dir = fs::path(data_path_) / "images";
    if (!fs::exists(images_dir) || !fs::is_directory(images_dir)) {
        std::cerr << "Warning: images directory not found: " << images_dir << std::endl;
    } else {
        // recursively collect images so split subfolders (train/val/test) are supported
        for (auto &p : fs::recursive_directory_iterator(images_dir)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                image_paths_.push_back(p.path().string());
            }
        }
    }

    // sort for deterministic order
    std::sort(image_paths_.begin(), image_paths_.end());

    // parse sequence-level label files into annotations_
    fs::path labels_dir = fs::path(data_path_) / "labels";
    if (fs::exists(labels_dir) && fs::is_directory(labels_dir)) {
        for (auto &p : fs::recursive_directory_iterator(labels_dir)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            if (ext == ".txt") {
                const std::string seq = p.path().stem().string();
                std::ifstream ifs(p.path());
                std::string line;
                while (std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream iss(line);
                    // Expected per-line format: <frame> <track_id> <class> <x> <y> <w> <h>
                    int frame = 0;
                    int track_id = 0;
                    int cls = 0;
                    float x = 0, y = 0, w = 0, h = 0;
                    if (!(iss >> frame >> track_id >> cls >> x >> y >> w >> h)) {
                        // malformed line; skip
                        continue;
                    }
                    Annotation ann;
                    ann.frame = frame;
                    ann.track_id = track_id;
                    ann.cls = cls;
                    ann.x = x;
                    ann.y = y;
                    ann.w = w;
                    ann.h = h;
                    annotations_[seq][frame].push_back(ann);
                }
            }
        }
    }
}

bool CAMELDataLoader::hasNextBatch() const {
    return current_index_ < image_paths_.size();
}

void CAMELDataLoader::reset() {
    current_index_ = 0;
}

static torch::Tensor matToTensor(const cv::Mat &img) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(640, 640));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // From HWC to CHW
    auto tensor = torch::from_blob(resized.data, {resized.rows, resized.cols, 3}, torch::kFloat32);
    tensor = tensor.permute({2, 0, 1}).contiguous().clone();
    return tensor;
}

cv::Mat CAMELDataLoader::preprocessImage(const cv::Mat &img) {
    // This is not used by matToTensor helper but kept for API completeness
    cv::Mat out;
    cv::resize(img, out, cv::Size(640, 640));
    return out;
}

bool CAMELDataLoader::parseImageFilename(const std::string &filename, std::string &seq, int &frame) const {
    fs::path p(filename);
    std::string stem = p.stem().string(); // e.g., Seq23_000440
    auto pos = stem.rfind('_');
    if (pos == std::string::npos) return false;
    seq = stem.substr(0, pos);
    std::string frame_str = stem.substr(pos + 1);
    try {
        frame = std::stoi(frame_str);
    } catch (...) {
        return false;
    }
    return true;
}

InfraredImageBatch CAMELDataLoader::getNextBatch() {
    std::vector<torch::Tensor> images;
    std::vector<torch::Tensor> target_rows;

    size_t count = 0;
    while (count < static_cast<size_t>(batch_size_) && hasNextBatch()) {
        const std::string img_path = image_paths_[current_index_];
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Failed to read image: " << img_path << std::endl;
            current_index_++;
            continue;
        }

        auto tensor_img = matToTensor(img);
        images.push_back(tensor_img);

        // For sequence-style datasets, parse the filename to extract sequence and frame
        fs::path p(img_path);
        std::string fname = p.filename().string();
        std::string seq;
        int frame = -1;
        if (parseImageFilename(fname, seq, frame)) {
            auto seq_it = annotations_.find(seq);
            if (seq_it != annotations_.end()) {
                auto frame_it = seq_it->second.find(frame);
                if (frame_it != seq_it->second.end()) {
                    // For each annotation on this frame, convert top-left to center and normalize
                    for (const auto &ann : frame_it->second) {
                        float cx = (ann.x + ann.w / 2.0f) / static_cast<float>(img.cols);
                        float cy = (ann.y + ann.h / 2.0f) / static_cast<float>(img.rows);
                        float wn = ann.w / static_cast<float>(img.cols);
                        float hn = ann.h / static_cast<float>(img.rows);
                        auto row = torch::tensor({static_cast<float>(count), static_cast<float>(ann.cls), cx, cy, wn, hn});
                        target_rows.push_back(row);
                    }
                }
            }
        }

        current_index_++;
        count++;
    }

    InfraredImageBatch batch;
    if (images.empty()) {
        batch.images = torch::empty({0});
    } else {
        batch.images = torch::stack(images);
    }

    if (target_rows.empty()) {
        batch.targets = torch::empty({0, 6});
    } else {
        batch.targets = torch::stack(target_rows);
    }

    return batch;
}
