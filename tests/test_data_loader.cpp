#include "../src/data_loader.cpp" // include implementation directly for simplicity in test
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;

int main() {
    const fs::path tmp = fs::path("tests") / "tmp_camel";
    const fs::path imgs = tmp / "images" / "train";
    const fs::path labels = tmp / "labels" / "train";
    fs::create_directories(imgs);
    fs::create_directories(labels);

    // write one synthetic image with size 480x640
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    const std::string img_path = (imgs / "Seq01_000001.jpg").string();
    cv::imwrite(img_path, img);

    // write corresponding sequence label: frame track_id class x y w h (absolute pixels)
    const std::string label_path = (labels / "Seq01.txt").string();
    std::ofstream ofs(label_path);
    // frame 1, track 10, class 0, top-left x=10,y=20,w=30,h=40
    ofs << "1 10 0 10 20 30 40\n";
    ofs.close();

    CAMELDataLoader loader(tmp.string(), 2);
    if (!loader.hasNextBatch()) {
        std::cerr << "Test failed: loader has no batches" << std::endl;
        return 2;
    }

    auto batch = loader.getNextBatch();
    std::cout << "Images tensor shape: " << batch.images.sizes() << std::endl;
    std::cout << "Targets tensor shape: " << batch.targets.sizes() << std::endl;

    if (batch.images.numel() == 0) {
        std::cerr << "Test failed: images empty" << std::endl;
        return 2;
    }
    if (batch.targets.numel() == 0) {
        std::cerr << "Test failed: targets empty" << std::endl;
        return 2;
    }

    // Verify the parsed target values (one annotation expected)
    // Expected normalized center/size values based on image size 640x480 and ann x=10,y=20,w=30,h=40
    const float expected_cx = (10.0f + 30.0f / 2.0f) / 640.0f; // 25/640
    const float expected_cy = (20.0f + 40.0f / 2.0f) / 480.0f; // 40/480
    const float expected_w = 30.0f / 640.0f;
    const float expected_h = 40.0f / 480.0f;

    auto targets = batch.targets;
    // pick first row
    auto row = targets[0];
    float batch_idx = row[0].item<float>();
    float cls = row[1].item<float>();
    float cx = row[2].item<float>();
    float cy = row[3].item<float>();
    float w = row[4].item<float>();
    float h = row[5].item<float>();

    const float eps = 1e-5f;
    if (std::abs(batch_idx - 0.0f) > eps) { std::cerr << "Bad batch_idx" << std::endl; return 2; }
    if (std::abs(cls - 0.0f) > eps) { std::cerr << "Bad class" << std::endl; return 2; }
    if (std::abs(cx - expected_cx) > 1e-4f) { std::cerr << "Bad cx: " << cx << std::endl; return 2; }
    if (std::abs(cy - expected_cy) > 1e-4f) { std::cerr << "Bad cy: " << cy << std::endl; return 2; }
    if (std::abs(w - expected_w) > 1e-5f) { std::cerr << "Bad w" << std::endl; return 2; }
    if (std::abs(h - expected_h) > 1e-5f) { std::cerr << "Bad h" << std::endl; return 2; }

    std::cout << "Test passed" << std::endl;
    return 0;
}
