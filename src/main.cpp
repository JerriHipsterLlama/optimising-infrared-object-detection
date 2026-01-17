#include <torch/torch.h>
#include "yolov8n.h"
#include "data_loader.h"
#include <iostream>
#include <fstream>

int main() {
    // Device setup (CPU or CUDA)
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;
    
    // Initialize model
    auto model = std::make_shared<YOLOv8n>();
    model->to(device);
    
    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), 
        torch::optim::SGDOptions(0.01).momentum(0.937));
    
    // Data loader
    CAMELDataLoader dataloader("data/camel", 32);
    
    int num_epochs = 100;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        dataloader.reset();
        double total_loss = 0.0;
        int batch_count = 0;
        
        while (dataloader.hasNextBatch()) {
            auto batch = dataloader.getNextBatch();
            batch.images = batch.images.to(device);
            batch.targets = batch.targets.to(device);
            
            // Forward pass
            auto predictions = model->forward(batch.images);
            
            // Compute loss (placeholder)
            auto loss = torch::nn::functional::mse_loss(predictions, batch.targets);
            
            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            total_loss += loss.item<double>();
            batch_count++;
        }
        
        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs 
                  << "] Loss: " << (total_loss / batch_count) << std::endl;
    }
    
    // Save model
    torch::save(model, "trained_models/yolov8n_camel.pt");
    std::cout << "Model saved to trained_models/yolov8n_camel.pt" << std::endl;
    
    return 0;
}