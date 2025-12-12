#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include "autoencoder_gpu.h"
#include "cifar10_loader.h"

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = cmd; \
        if (e != cudaSuccess) { \
            std::cout << "CUDA Error " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(e) << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {
    std::cout << "=== CIFAR-10 Autoencoder Training (GPU) ===\n\n";

    // 1. Load CIFAR-10 data
    std::string data_dir = "data/cifar-10-batches-bin";
    CIFAR10Loader loader(data_dir);

    std::vector<float> train_images;
    std::vector<int> train_labels;
    std::cout << "Loading training data...\n";
    loader.load_train_data(train_images, train_labels);
    std::cout << "Loaded " << train_images.size() / CIFAR10Loader::IMAGE_SIZE 
              << " training images\n\n";

    // 2. Setup training
    int batch_size = 64;  // Theo đề bài
    int H = 32, W = 32;
    int epochs = 20;      // Theo đề bài
    float lr = 1e-3f;

    int num_train = train_images.size() / CIFAR10Loader::IMAGE_SIZE;
    int num_batches = (num_train + batch_size - 1) / batch_size;

    std::cout << "Training configuration:\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Epochs: " << epochs << "\n";
    std::cout << "  Learning rate: " << lr << "\n";
    std::cout << "  Batches per epoch: " << num_batches << "\n\n";

    // 3. Initialize AutoencoderGPU
    AutoencoderGPU ae(batch_size, H, W, lr);

    // 4. Allocate GPU memory for batch
    int batch_image_size = batch_size * CIFAR10Loader::IMAGE_SIZE;
    float *d_batch_input = nullptr;
    float *d_batch_recon = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_recon, batch_image_size * sizeof(float)));

    // 5. Training loop
    std::cout << "Starting training...\n";
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data mỗi epoch
        loader.shuffle_data(train_images, train_labels);

        auto start_epoch = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;
        int batches_processed = 0;

        // Train trên tất cả batches
        for (int b = 0; b < num_batches; ++b) {
            int start_idx = b * batch_size;
            int current_batch_size = std::min(batch_size, num_train - start_idx);
            
            // Get batch
            std::vector<float> batch_images;
            std::vector<int> batch_labels;  // Không dùng cho autoencoder nhưng cần cho get_batch
            loader.get_batch(train_images, train_labels,
                           batch_images, batch_labels,
                           current_batch_size, start_idx);

            // Copy batch to GPU (only current_batch_size * IMAGE_SIZE elements)
            CUDA_CHECK(cudaMemcpy(d_batch_input, batch_images.data(),
                                 batch_images.size() * sizeof(float),
                                 cudaMemcpyHostToDevice));

            // Train step: PASS actual batch size
            float loss = ae.train_step(d_batch_input, d_batch_recon, current_batch_size);
            epoch_loss += loss;
            batches_processed++;
        }

        auto end_epoch = std::chrono::high_resolution_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_epoch - start_epoch).count();

        float avg_loss = epoch_loss / batches_processed;
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " - Loss: " << avg_loss
                  << " - Time: " << epoch_time << " ms\n";
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        end_total - start_total).count();

    std::cout << "\n=== Training Complete ===\n";
    std::cout << "Total training time: " << total_time << " seconds ("
              << total_time / 60.0 << " minutes)\n";

    // 6. Save trained weights
    std::string weights_file = "models/autoencoder_weights.bin";
    std::cout << "\nSaving trained weights to " << weights_file << "..." << std::endl;
    // Tạo folder models nếu chưa có
    int mkdir_result = system("mkdir -p models");
    (void)mkdir_result;  // Suppress unused warning
    ae.save_weights(weights_file);
    std::cout << "✓ Weights saved successfully!\n";

    // 7. Cleanup
    CUDA_CHECK(cudaFree(d_batch_input));
    CUDA_CHECK(cudaFree(d_batch_recon));

    return 0;
}