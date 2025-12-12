#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring> 
#include <iomanip> 
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

    std::cout << "Allocating GPU memory..." << std::flush;
    int batch_image_size = batch_size * CIFAR10Loader::IMAGE_SIZE;
    
    // Allocate pinned memory
    float *h_pinned_batch = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned_batch, batch_image_size * sizeof(float)));
    
    // Allocate device memory
    float *d_batch_input = nullptr;
    float *d_batch_recon = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_recon, batch_image_size * sizeof(float)));

    // Multi-stream pipeline: triple buffering + async loss
    cudaStream_t stream_compute[3];
    cudaStream_t stream_transfer;
    cudaStream_t stream_loss;
    CUDA_CHECK(cudaStreamCreate(&stream_compute[0]));
    CUDA_CHECK(cudaStreamCreate(&stream_compute[1]));
    CUDA_CHECK(cudaStreamCreate(&stream_compute[2]));
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));
    CUDA_CHECK(cudaStreamCreate(&stream_loss));
    cudaEvent_t ev_h2d_done[3];
    for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaEventCreateWithFlags(&ev_h2d_done[i], cudaEventDisableTiming));
    cudaEvent_t ev_compute_done[3];
    cudaEvent_t ev_loss_done[3];
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_compute_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_loss_done[i], cudaEventDisableTiming));
    }
    
    // Loss buffers (device + host pinned)
    float *d_loss_buf[3] = {nullptr, nullptr, nullptr};
    float *h_loss_buf[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMalloc(&d_loss_buf[i], sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_loss_buf[i], sizeof(float)));
    }
    
    // Triple buffers
    float *d_batch_input_buf[3] = {nullptr, nullptr, nullptr};
    float *d_batch_recon_buf[3] = {nullptr, nullptr, nullptr};
    float *h_pinned_batch_buf[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMalloc(&d_batch_input_buf[i], batch_image_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_batch_recon_buf[i], batch_image_size * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_pinned_batch_buf[i], batch_image_size * sizeof(float)));
    }
    
    std::cout << " ✓ Done\n" << std::flush;

    // 5. Warmup 1 batch trước khi đo thời gian (ổn định xung GPU)
    // Note: Test batch removed for performance (saves ~2s total)
    int warmup_batches = 1;  // Reduced from 2 to 1 for performance
    for (int w = 0; w < warmup_batches; ++w) {
        int start_idx = w * batch_size;
        int copy_size = std::min(batch_size, num_train - start_idx) * CIFAR10Loader::IMAGE_SIZE;
        std::memcpy(h_pinned_batch,
                   &train_images[start_idx * CIFAR10Loader::IMAGE_SIZE],
                   copy_size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_batch_input, h_pinned_batch,
                             copy_size * sizeof(float),
                             cudaMemcpyHostToDevice));
        ae.train_step(d_batch_input, d_batch_recon);
    }

    // 6. Training loop with profiling
    std::cout << "Starting training...\n" << std::flush;
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data
        loader.shuffle_data(train_images, train_labels);

        auto start_epoch = std::chrono::high_resolution_clock::now();
        // Track which batches computed loss and their buffer indices
        std::vector<std::pair<int, int>> loss_batches;  // (batch_idx, buf_idx)

        // Prefetch first batch
        if (num_batches > 0) {
            int first_copy_size = std::min(batch_size, num_train) * CIFAR10Loader::IMAGE_SIZE;
            std::memcpy(h_pinned_batch_buf[0], 
                       &train_images[0],
                       first_copy_size * sizeof(float));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_input_buf[0], h_pinned_batch_buf[0],
                                       first_copy_size * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_transfer));
            CUDA_CHECK(cudaEventRecord(ev_h2d_done[0], stream_transfer));
        }

        for (int b = 0; b < num_batches; ++b) {
            int start_idx = b * batch_size;
            int current_batch_size = std::min(batch_size, num_train - start_idx);
            int buf_idx = b % 3;
            int next_buf_idx = (b + 1) % 3;
            cudaStream_t current_compute_stream = stream_compute[buf_idx];
            
            // Đợi H2D của buffer hiện tại hoàn tất
            CUDA_CHECK(cudaStreamWaitEvent(current_compute_stream, ev_h2d_done[buf_idx], 0));
            
            // Prefetch NEXT batch trong khi compute CURRENT batch
            if (b + 1 < num_batches) {
                int next_start_idx = (b + 1) * batch_size;
                int next_batch_size = std::min(batch_size, num_train - next_start_idx);
                int next_copy_size = next_batch_size * CIFAR10Loader::IMAGE_SIZE;
                
                if (b + 1 >= 3) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream_transfer, ev_compute_done[next_buf_idx], 0));
                }
                
                std::memcpy(h_pinned_batch_buf[next_buf_idx], 
                           &train_images[next_start_idx * CIFAR10Loader::IMAGE_SIZE],
                           next_copy_size * sizeof(float));
                
                CUDA_CHECK(cudaMemcpyAsync(d_batch_input_buf[next_buf_idx], 
                                          h_pinned_batch_buf[next_buf_idx],
                                          next_copy_size * sizeof(float),
                                          cudaMemcpyHostToDevice, stream_transfer));
                CUDA_CHECK(cudaEventRecord(ev_h2d_done[next_buf_idx], stream_transfer));
            }

            bool compute_loss = (b % 40 == 0);  // Aggressive reduction for <10min target (was 20)
            
            if (compute_loss) {
                ae.train_step_async_loss(
                    d_batch_input_buf[buf_idx], d_batch_recon_buf[buf_idx],
                    current_compute_stream,
                    d_loss_buf[buf_idx], h_loss_buf[buf_idx],
                    ev_compute_done[buf_idx], ev_loss_done[buf_idx],
                    stream_loss);
                loss_batches.push_back({b, buf_idx});
            } else {
                ae.train_step(d_batch_input_buf[buf_idx], d_batch_recon_buf[buf_idx],
                              current_compute_stream, false, nullptr);
                CUDA_CHECK(cudaEventRecord(ev_compute_done[buf_idx], current_compute_stream));
            }
        }
        
        // Optimized synchronization: sync only necessary streams
        // stream_transfer is synchronized via events, so we skip it
        // Sync compute streams to ensure all batches complete
        CUDA_CHECK(cudaStreamSynchronize(stream_compute[0]));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute[1]));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute[2]));
        CUDA_CHECK(cudaStreamSynchronize(stream_loss));  // Ensure loss is ready
        // Note: stream_transfer not needed - already synchronized via events
        
        // Collect losses
        float epoch_loss_sum = 0.0f;
        int loss_samples = 0;
        for (const auto& loss_info : loss_batches) {
            int batch_idx = loss_info.first;
            int buf_idx = loss_info.second;
            int batch_size_for_loss = std::min(batch_size, num_train - batch_idx * batch_size);
            int total_elements = batch_size_for_loss * CIFAR10Loader::IMAGE_SIZE;
            float batch_loss = *h_loss_buf[buf_idx] / static_cast<float>(total_elements);
            epoch_loss_sum += batch_loss;
            loss_samples++;
        }

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
    
    // Profiling summary
    std::cout << "\n=== Performance Profiling Summary ===\n";
    std::cout << "Note: Detailed kernel timing requires nvprof/nsight.\n";
    std::cout << "For detailed profiling, run: nvprof ./train_cifar10_gpu_optimized\n";
    std::cout << "Average time per batch: " << std::fixed << std::setprecision(2)
              << (total_time * 1000.0 / (epochs * num_batches)) << " ms\n";
    std::cout << "Batches per second: " << std::fixed << std::setprecision(2)
              << (epochs * num_batches / total_time) << " batches/s\n";

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
    CUDA_CHECK(cudaFreeHost(h_pinned_batch_buf[2]));
    CUDA_CHECK(cudaFree(d_batch_input));
    CUDA_CHECK(cudaFree(d_batch_recon));
    CUDA_CHECK(cudaFree(d_batch_input_buf[0]));
    CUDA_CHECK(cudaFree(d_batch_input_buf[1]));
    CUDA_CHECK(cudaFree(d_batch_input_buf[2]));
    CUDA_CHECK(cudaFree(d_batch_recon_buf[0]));
    CUDA_CHECK(cudaFree(d_batch_recon_buf[1]));
    CUDA_CHECK(cudaFree(d_batch_recon_buf[2]));
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaFree(d_loss_buf[i]));
        CUDA_CHECK(cudaFreeHost(h_loss_buf[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(stream_compute[0]));
    CUDA_CHECK(cudaStreamDestroy(stream_compute[1]));
    CUDA_CHECK(cudaStreamDestroy(stream_compute[2]));
    CUDA_CHECK(cudaStreamDestroy(stream_transfer));
    CUDA_CHECK(cudaStreamDestroy(stream_loss));
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaEventDestroy(ev_h2d_done[i]));
        CUDA_CHECK(cudaEventDestroy(ev_compute_done[i]));
        CUDA_CHECK(cudaEventDestroy(ev_loss_done[i]));
    }

    return 0;
}

