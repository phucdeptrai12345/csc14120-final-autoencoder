#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
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
    std::cout << "=== CIFAR-10 Feature Extraction (GPU) ===\n\n";

    // 1. Load CIFAR-10 data
    std::string data_dir = "data/cifar-10-batches-bin";
    CIFAR10Loader loader(data_dir);

    std::vector<float> train_images, test_images;
    std::vector<int> train_labels, test_labels;
    
    std::cout << "Loading training data...\n";
    loader.load_train_data(train_images, train_labels);
    std::cout << "Loaded " << train_images.size() / CIFAR10Loader::IMAGE_SIZE 
              << " training images\n\n";

    std::cout << "Loading test data...\n";
    loader.load_test_data(test_images, test_labels);
    std::cout << "Loaded " << test_images.size() / CIFAR10Loader::IMAGE_SIZE 
              << " test images\n\n";

    // 2. Initialize AutoencoderGPU và load trained weights
    int batch_size = 64;
    int H = 32, W = 32;
    float lr = 1e-3f;  // Không dùng cho extraction nhưng cần cho constructor
    
    std::cout << "Initializing AutoencoderGPU...\n";
    AutoencoderGPU ae(batch_size, H, W, lr);
    
    std::string weights_file = "models/autoencoder_weights.bin";
    std::cout << "Loading trained weights from " << weights_file << "...\n";
    ae.load_weights(weights_file);
    std::cout << "✓ Weights loaded successfully!\n\n";

    // 3. Extract features cho training set
    int num_train = train_images.size() / CIFAR10Loader::IMAGE_SIZE;
    int num_test = test_images.size() / CIFAR10Loader::IMAGE_SIZE;
    int feature_dim = 128 * 8 * 8;  // 8192 dimensions
    
    std::cout << "Extracting features for training set (" << num_train << " images)...\n";
    auto start_train = std::chrono::high_resolution_clock::now();
    
    std::vector<float> train_features(num_train * feature_dim);
    float *d_batch_input = nullptr;
    float *d_batch_features = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_size * CIFAR10Loader::IMAGE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_features, batch_size * feature_dim * sizeof(float)));
    
    int num_batches_train = (num_train + batch_size - 1) / batch_size;
    for (int b = 0; b < num_batches_train; ++b) {
        int start_idx = b * batch_size;
        int actual_batch_size = std::min(batch_size, num_train - start_idx);
        
        // Get batch
        std::vector<float> batch_images;
        std::vector<int> batch_labels;
        loader.get_batch(train_images, train_labels,
                        batch_images, batch_labels,
                        actual_batch_size, start_idx);
        
        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(d_batch_input, batch_images.data(),
                             batch_images.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
        
        // Extract features
        ae.extract_features(d_batch_input, d_batch_features);
        
        // Copy back to host
        CUDA_CHECK(cudaMemcpy(&train_features[start_idx * feature_dim],
                             d_batch_features,
                             actual_batch_size * feature_dim * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        if ((b + 1) % 100 == 0) {
            std::cout << "  Processed " << (b + 1) << "/" << num_batches_train << " batches\n";
        }
    }
    
    auto end_train = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_train - start_train).count();
    std::cout << "✓ Training features extracted in " << train_time << " ms\n\n";

    // 4. Extract features cho test set
    std::cout << "Extracting features for test set (" << num_test << " images)...\n";
    auto start_test = std::chrono::high_resolution_clock::now();
    
    std::vector<float> test_features(num_test * feature_dim);
    int num_batches_test = (num_test + batch_size - 1) / batch_size;
    
    for (int b = 0; b < num_batches_test; ++b) {
        int start_idx = b * batch_size;
        int actual_batch_size = std::min(batch_size, num_test - start_idx);
        
        // Get batch
        std::vector<float> batch_images;
        std::vector<int> batch_labels;
        loader.get_batch(test_images, test_labels,
                        batch_images, batch_labels,
                        actual_batch_size, start_idx);
        
        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(d_batch_input, batch_images.data(),
                             batch_images.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
        
        // Extract features
        ae.extract_features(d_batch_input, d_batch_features);
        
        // Copy back to host
        CUDA_CHECK(cudaMemcpy(&test_features[start_idx * feature_dim],
                             d_batch_features,
                             actual_batch_size * feature_dim * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        if ((b + 1) % 50 == 0) {
            std::cout << "  Processed " << (b + 1) << "/" << num_batches_test << " batches\n";
        }
    }
    
    auto end_test = std::chrono::high_resolution_clock::now();
    auto test_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_test - start_test).count();
    std::cout << "✓ Test features extracted in " << test_time << " ms\n\n";

    // 5. Save features to files (format: LIBSVM compatible hoặc binary)
    std::cout << "Saving features to files...\n";
    
    // Save training features (binary format: num_images, feature_dim, features, labels)
    std::string train_feat_file = "models/train_features.bin";
    std::ofstream train_file(train_feat_file, std::ios::binary);
    train_file.write(reinterpret_cast<const char*>(&num_train), sizeof(int));
    train_file.write(reinterpret_cast<const char*>(&feature_dim), sizeof(int));
    train_file.write(reinterpret_cast<const char*>(train_features.data()),
                    train_features.size() * sizeof(float));
    train_file.write(reinterpret_cast<const char*>(train_labels.data()),
                    train_labels.size() * sizeof(int));
    train_file.close();
    std::cout << "✓ Saved training features to " << train_feat_file << "\n";
    
    // Save test features
    std::string test_feat_file = "models/test_features.bin";
    std::ofstream test_file(test_feat_file, std::ios::binary);
    test_file.write(reinterpret_cast<const char*>(&num_test), sizeof(int));
    test_file.write(reinterpret_cast<const char*>(&feature_dim), sizeof(int));
    test_file.write(reinterpret_cast<const char*>(test_features.data()),
                   test_features.size() * sizeof(float));
    test_file.write(reinterpret_cast<const char*>(test_labels.data()),
                   test_labels.size() * sizeof(int));
    test_file.close();
    std::cout << "✓ Saved test features to " << test_feat_file << "\n\n";

    // 6. Summary
    std::cout << "=== Feature Extraction Complete ===\n";
    std::cout << "Training features: " << num_train << " x " << feature_dim << " = "
              << (num_train * feature_dim * sizeof(float) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Test features: " << num_test << " x " << feature_dim << " = "
              << (num_test * feature_dim * sizeof(float) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Total extraction time: " << (train_time + test_time) / 1000.0 << " seconds\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_batch_input));
    CUDA_CHECK(cudaFree(d_batch_features));

    return 0;
}

