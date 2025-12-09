#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <cstring>  // for memcpy
#include "autoencoder_gpu_optimized.h"
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
    std::cout << "=== CIFAR-10 Feature Extraction (GPU OPTIMIZED) ===\n\n";

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

    // 2. Initialize AutoencoderGPUOptimized và load trained weights
    // OPTIMIZATION: Use larger batch size for feature extraction (no backward pass needed)
    int batch_size = 128;  // Larger than training (64) for better GPU utilization
    int H = 32, W = 32;
    float lr = 1e-3f;  // Not used for extraction but needed for constructor
    
    std::cout << "Initializing AutoencoderGPUOptimized...\n";
    AutoencoderGPUOptimized ae(batch_size, H, W, lr);
    
    std::string weights_file = "models/autoencoder_weights.bin";
    std::cout << "Loading trained weights from " << weights_file << "...\n";
    ae.load_weights(weights_file);
    std::cout << "✓ Weights loaded successfully!\n\n";

    // 3. Create CUDA streams for async operations
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    // 4. Allocate pinned memory for faster H2D transfers
    int num_train = train_images.size() / CIFAR10Loader::IMAGE_SIZE;
    int num_test = test_images.size() / CIFAR10Loader::IMAGE_SIZE;
    int feature_dim = 128 * 8 * 8;  // 8192 dimensions
    
    size_t batch_image_size = batch_size * CIFAR10Loader::IMAGE_SIZE;
    size_t batch_feature_size = batch_size * feature_dim;
    
    float *h_pinned_batch = nullptr;
    float *d_batch_input = nullptr;
    float *d_batch_features = nullptr;
    
    CUDA_CHECK(cudaMallocHost(&h_pinned_batch, batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_features, batch_feature_size * sizeof(float)));

    // 5. Extract features cho training set
    std::cout << "Extracting features for training set (" << num_train << " images)...\n";
    auto start_train = std::chrono::high_resolution_clock::now();
    
    std::vector<float> train_features(num_train * feature_dim);
    int num_batches_train = (num_train + batch_size - 1) / batch_size;
    
    for (int b = 0; b < num_batches_train; ++b) {
        int start_idx = b * batch_size;
        int actual_batch_size = std::min(batch_size, num_train - start_idx);
        int copy_size = actual_batch_size * CIFAR10Loader::IMAGE_SIZE;
        
        // OPTIMIZATION: Use pinned memory and async transfer
        cudaStream_t current_stream = (b % 2 == 0) ? stream1 : stream2;
        
        // Copy batch to pinned memory (CPU-side, fast)
        std::memcpy(h_pinned_batch, &train_images[start_idx * CIFAR10Loader::IMAGE_SIZE],
                   copy_size * sizeof(float));
        
        // Async H2D transfer (overlaps with previous batch processing if possible)
        CUDA_CHECK(cudaMemcpyAsync(d_batch_input, h_pinned_batch,
                                 copy_size * sizeof(float),
                                 cudaMemcpyHostToDevice, current_stream));
        
        // Extract features (uses optimized fused kernels)
        ae.extract_features(d_batch_input, d_batch_features, current_stream);
        
        // Async D2H transfer
        CUDA_CHECK(cudaMemcpyAsync(&train_features[start_idx * feature_dim],
                                 d_batch_features,
                                 actual_batch_size * feature_dim * sizeof(float),
                                 cudaMemcpyDeviceToHost, current_stream));
        
        // Sync stream to ensure data is ready (can be optimized further with double buffering)
        CUDA_CHECK(cudaStreamSynchronize(current_stream));
        
        if ((b + 1) % 100 == 0) {
            std::cout << "  Processed " << (b + 1) << "/" << num_batches_train << " batches\r";
            std::cout.flush();
        }
    }
    
    auto end_train = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_train - start_train).count();
    std::cout << "\n✓ Training features extracted in " << train_time << " ms ("
              << (train_time / 1000.0) << " seconds)\n\n";

    // 6. Extract features cho test set
    std::cout << "Extracting features for test set (" << num_test << " images)...\n";
    auto start_test = std::chrono::high_resolution_clock::now();
    
    std::vector<float> test_features(num_test * feature_dim);
    int num_batches_test = (num_test + batch_size - 1) / batch_size;
    
    for (int b = 0; b < num_batches_test; ++b) {
        int start_idx = b * batch_size;
        int actual_batch_size = std::min(batch_size, num_test - start_idx);
        int copy_size = actual_batch_size * CIFAR10Loader::IMAGE_SIZE;
        
        cudaStream_t current_stream = (b % 2 == 0) ? stream1 : stream2;
        
        std::memcpy(h_pinned_batch, &test_images[start_idx * CIFAR10Loader::IMAGE_SIZE],
                   copy_size * sizeof(float));
        
        CUDA_CHECK(cudaMemcpyAsync(d_batch_input, h_pinned_batch,
                                 copy_size * sizeof(float),
                                 cudaMemcpyHostToDevice, current_stream));
        
        ae.extract_features(d_batch_input, d_batch_features, current_stream);
        
        CUDA_CHECK(cudaMemcpyAsync(&test_features[start_idx * feature_dim],
                                 d_batch_features,
                                 actual_batch_size * feature_dim * sizeof(float),
                                 cudaMemcpyDeviceToHost, current_stream));
        
        CUDA_CHECK(cudaStreamSynchronize(current_stream));
        
        if ((b + 1) % 50 == 0) {
            std::cout << "  Processed " << (b + 1) << "/" << num_batches_test << " batches\r";
            std::cout.flush();
        }
    }
    
    auto end_test = std::chrono::high_resolution_clock::now();
    auto test_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_test - start_test).count();
    std::cout << "\n✓ Test features extracted in " << test_time << " ms ("
              << (test_time / 1000.0) << " seconds)\n\n";

    // 7. Save features to files
    std::cout << "Saving features to files...\n";
    
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

    // 8. Summary
    double total_time = (train_time + test_time) / 1000.0;
    std::cout << "=== Feature Extraction Complete ===\n";
    std::cout << "Training features: " << num_train << " x " << feature_dim << " = "
              << (num_train * feature_dim * sizeof(float) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Test features: " << num_test << " x " << feature_dim << " = "
              << (num_test * feature_dim * sizeof(float) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Total extraction time: " << total_time << " seconds\n";
    std::cout << "Target: <20 seconds\n";
    if (total_time < 20.0) {
        std::cout << "✓ TARGET ACHIEVED! (" << total_time << "s < 20s)\n";
    } else {
        std::cout << "⚠ Target not met (" << total_time << "s >= 20s)\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_pinned_batch));
    CUDA_CHECK(cudaFree(d_batch_input));
    CUDA_CHECK(cudaFree(d_batch_features));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}

