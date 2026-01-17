#pragma once
#include <torch/torch.h>

class YOLOv8nImpl : public torch::nn::Module {
public:
    YOLOv8nImpl();
    
    torch::Tensor forward(torch::Tensor x);
    
private:
    // Backbone layers
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    
    // Detection head layers
    torch::nn::Conv2d detect_head{nullptr};
};

TORCH_MODULE(YOLOv8n);