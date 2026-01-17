#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <unordered_map>

struct Annotation {
    int frame;
    int track_id;
    int cls;
    float x; // top-left x
    float y; // top-left y
    float w;
    float h;
};
struct InfraredImageBatch {
    torch::Tensor images;  // Shape: [batch_size, 3, 640, 640]
    torch::Tensor targets; // Shape: [num_objects, 6] (batch_idx, class, x, y, w, h)
};

class CAMELDataLoader {
public:
    CAMELDataLoader(const std::string& data_path, int batch_size);
    
    InfraredImageBatch getNextBatch();
    bool hasNextBatch() const;
    void reset();
    
private:
    std::string data_path_;
    int batch_size_;
    std::vector<std::string> image_paths_;
    size_t current_index_ = 0;
    // sequence -> (frame -> annotations)
    std::unordered_map<std::string, std::unordered_map<int, std::vector<Annotation>>> annotations_;

    bool parseImageFilename(const std::string &filename, std::string &seq, int &frame) const;
    
    cv::Mat preprocessImage(const cv::Mat& img);
};