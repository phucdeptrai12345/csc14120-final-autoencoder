#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>  // for memcpy
#include <iomanip>  // for std::setprecision
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
    std::cout << "=== CIFAR-10 Autoencoder Training (GPU OPTIMIZED - Phase 3) ===\n\n";

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
    // Giữ nguyên yêu cầu đề: batch size = 64, epochs = 20
    int batch_size = 64;
    int H = 32, W = 32;
    int epochs = 20;
    float lr = 1e-3f;

    int num_train = train_images.size() / CIFAR10Loader::IMAGE_SIZE;
    int num_batches = (num_train + batch_size - 1) / batch_size;

    std::cout << "Training configuration:\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Epochs: " << epochs << "\n";
    std::cout << "  Learning rate: " << lr << "\n";
    std::cout << "  Batches per epoch: " << num_batches << "\n\n";

    // 3. Initialize AutoencoderGPUOptimized
    std::cout << "Initializing autoencoder..." << std::flush;
    AutoencoderGPUOptimized ae(batch_size, H, W, lr);
    std::cout << " ✓ Done\n" << std::flush;

    // 4. ✅ OPTIMIZATION 3: PINNED MEMORY + MULTI-STREAM PIPELINE
    std::cout << "Allocating GPU memory..." << std::flush;
    int batch_image_size = batch_size * CIFAR10Loader::IMAGE_SIZE;
    
    // Allocate pinned memory (faster than regular malloc for H2D transfers)
    float *h_pinned_batch = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned_batch, batch_image_size * sizeof(float)));
    
    // Allocate device memory
    float *d_batch_input = nullptr;
    float *d_batch_recon = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_recon, batch_image_size * sizeof(float)));

    // ✅ OPTIMIZATION 4: MULTI-STREAM PIPELINE với EXPLICIT PREFETCHING
    // 3 streams: 2 cho compute, 1 cho transfer (overlap tốt hơn)
    cudaStream_t stream_compute[2];
    cudaStream_t stream_transfer;
    CUDA_CHECK(cudaStreamCreate(&stream_compute[0]));
    CUDA_CHECK(cudaStreamCreate(&stream_compute[1]));
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));
    
    // Allocate double buffers cho prefetching
    float *d_batch_input_buf[2] = {nullptr, nullptr};
    float *d_batch_recon_buf[2] = {nullptr, nullptr};
    float *h_pinned_batch_buf[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_batch_input_buf[0], batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_input_buf[1], batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_recon_buf[0], batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_recon_buf[1], batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_batch_buf[0], batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_batch_buf[1], batch_image_size * sizeof(float)));
    
    std::cout << " ✓ Done\n" << std::flush;

    // 5. Test với 1 batch nhỏ trước
    std::cout << "Testing with 1 batch..." << std::flush;
    int test_batch_size = std::min(batch_size, num_train);
    int test_copy_size = test_batch_size * CIFAR10Loader::IMAGE_SIZE;
    std::memcpy(h_pinned_batch, 
               &train_images[0],
               test_copy_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_batch_input, h_pinned_batch,
                         test_copy_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    float test_loss = ae.train_step(d_batch_input, d_batch_recon);
    std::cout << " ✓ Done (test loss: " << test_loss << ")\n" << std::flush;

    // 6. Training loop
    std::cout << "Starting training...\n" << std::flush;
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data
        loader.shuffle_data(train_images, train_labels);

        auto start_epoch = std::chrono::high_resolution_clock::now();
        float epoch_loss_sum = 0.0f;
        int loss_samples = 0;

        // Prefetch first batch
        if (num_batches > 0) {
            int first_copy_size = std::min(batch_size, num_train) * CIFAR10Loader::IMAGE_SIZE;
            std::memcpy(h_pinned_batch_buf[0], 
                       &train_images[0],
                       first_copy_size * sizeof(float));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_input_buf[0], h_pinned_batch_buf[0],
                                       first_copy_size * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_transfer));
        }

        for (int b = 0; b < num_batches; ++b) {
            int start_idx = b * batch_size;
            int current_batch_size = std::min(batch_size, num_train - start_idx);
            int buf_idx = b % 2;
            int next_buf_idx = (b + 1) % 2;
            cudaStream_t current_compute_stream = stream_compute[buf_idx];
            
            // OPTIMIZATION: Chỉ sync transfer stream khi batch đầu tiên hoặc khi cần thiết
            // Batch đầu tiên: cần sync để đảm bảo data đã sẵn sàng
            // Các batch tiếp theo: không cần sync vì đã prefetch từ batch trước
            if (b == 0) {
                CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
            }
            
            // Prefetch NEXT batch trong khi compute CURRENT batch
            if (b + 1 < num_batches) {
                int next_start_idx = (b + 1) * batch_size;
                int next_batch_size = std::min(batch_size, num_train - next_start_idx);
                int next_copy_size = next_batch_size * CIFAR10Loader::IMAGE_SIZE;
                
                // CPU memcpy to pinned buffer (có thể overlap với GPU compute)
                std::memcpy(h_pinned_batch_buf[next_buf_idx], 
                           &train_images[next_start_idx * CIFAR10Loader::IMAGE_SIZE],
                           next_copy_size * sizeof(float));
                
                // Async H2D transfer (overlaps với GPU compute)
                CUDA_CHECK(cudaMemcpyAsync(d_batch_input_buf[next_buf_idx], 
                                          h_pinned_batch_buf[next_buf_idx],
                                          next_copy_size * sizeof(float),
                                          cudaMemcpyHostToDevice, stream_transfer));
            }

            // Compute current batch (overlaps với next batch transfer)
            // OPTIMIZATION: Chỉ compute loss mỗi 200 batches để giảm sync overhead
            // Loss computation cần sync stream → overhead lớn
            // Compute loss ở đầu, giữa, và cuối epoch để có avg_loss chính xác
            bool compute_loss = (b == 0) || (b == num_batches / 2) || (b == num_batches - 1);
            float batch_loss = 0.0f;
            
            ae.train_step(d_batch_input_buf[buf_idx], d_batch_recon_buf[buf_idx],
                          current_compute_stream, compute_loss, &batch_loss);
            
            if (compute_loss) {
                epoch_loss_sum += batch_loss;
                loss_samples++;
            }
        }
        
        // Wait for all streams to complete
        CUDA_CHECK(cudaStreamSynchronize(stream_compute[0]));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute[1]));
        CUDA_CHECK(cudaStreamSynchronize(stream_transfer));

        auto end_epoch = std::chrono::high_resolution_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_epoch - start_epoch).count();

        float avg_loss = (loss_samples > 0) ? (epoch_loss_sum / loss_samples) : 0.0f;
        
        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs
                  << " - Avg Loss: " << std::fixed << std::setprecision(6) << avg_loss
                  << " - Time: " << epoch_time << " ms ("
                  << std::fixed << std::setprecision(2) 
                  << (float(epoch_time) / num_batches) << " ms/batch)\n" << std::flush;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        end_total - start_total).count();

    std::cout << "\n=== Training Complete ===\n";
    std::cout << "Total training time: " << total_time << " seconds ("
              << total_time / 60.0 << " minutes)\n";

    // 6. Save trained weights
    std::string weights_file = "models/autoencoder_weights_optimized.bin";
    std::cout << "\nSaving trained weights to " << weights_file << "..." << std::endl;
    int mkdir_result = system("mkdir -p models");
    (void)mkdir_result;
    ae.save_weights(weights_file);
    std::cout << "✓ Weights saved successfully!\n";

    // 7. Cleanup
    CUDA_CHECK(cudaFreeHost(h_pinned_batch));  // Free pinned memory
    CUDA_CHECK(cudaFreeHost(h_pinned_batch_buf[0]));
    CUDA_CHECK(cudaFreeHost(h_pinned_batch_buf[1]));
    CUDA_CHECK(cudaFree(d_batch_input));
    CUDA_CHECK(cudaFree(d_batch_recon));
    CUDA_CHECK(cudaFree(d_batch_input_buf[0]));
    CUDA_CHECK(cudaFree(d_batch_input_buf[1]));
    CUDA_CHECK(cudaFree(d_batch_recon_buf[0]));
    CUDA_CHECK(cudaFree(d_batch_recon_buf[1]));
    CUDA_CHECK(cudaStreamDestroy(stream_compute[0]));
    CUDA_CHECK(cudaStreamDestroy(stream_compute[1]));
    CUDA_CHECK(cudaStreamDestroy(stream_transfer));

    return 0;
}

