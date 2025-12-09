#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "layers_gpu.h"
#include "autoencoder_gpu.h"

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = cmd; \
        if (e != cudaSuccess) { \
            std::cout << "CUDA Error " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(e) << std::endl; \
        } \
    } while (0)

int main() {
    std::cout << "Running AutoencoderGPU training test (CIFAR-10 architecture)...\n";

    // Architecture theo spec: 32x32x3 (CIFAR-10)
    int N = 1;  // batch size
    int H = 32, W = 32;
    int C = 3;  // RGB channels

    // Tạo input giả lập (32x32x3) - random values [0,1]
    int in_size = N * C * H * W;
    std::vector<float> h_input(in_size);
    for (int i = 0; i < in_size; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // [0,1]
    }

    float *d_input = nullptr;
    float *d_recon = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, in_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_recon, in_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          in_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Khởi tạo AutoencoderGPU (architecture cố định: 3→256→128→128→256→3)
    float lr = 1e-3f;
    AutoencoderGPU ae(N, H, W, lr);

    // Train
    // - Test nhanh: 20-30 epochs
    // - Test đầy đủ: 50-100 epochs  
    // - Train thật (CIFAR-10): 20 epochs (theo đề bài)
    int epochs = 30;  // Tăng lên 50-100 nếu muốn train kỹ hơn
    std::cout << "Training for " << epochs << " epochs...\n";
    std::cout << "Loss sẽ được in mỗi 5 epochs\n";
    
    for (int e = 0; e < epochs; ++e) {
        float loss = ae.train_step(d_input, d_recon);
        // In loss mỗi 5 epochs hoặc epoch cuối
        if ((e + 1) % 5 == 0 || e == 0 || e == epochs - 1) {
            std::cout << "Epoch " << (e + 1) << "/" << epochs
                      << " - loss: " << loss << "\n";
        }
    }

    // Test feature extraction
    std::cout << "\nTesting feature extraction...\n";
    int latent_size = ae.latent_size();
    float *d_features = nullptr;
    CUDA_CHECK(cudaMalloc(&d_features, latent_size * sizeof(float)));
    ae.extract_features(d_input, d_features);
    std::cout << "Feature extraction successful! Latent size: " << latent_size
              << " (should be " << N << " * 128 * 8 * 8 = " << (N * 128 * 8 * 8) << ")\n";

    // Copy output về host (chỉ in một vài giá trị để kiểm tra)
    std::vector<float> h_recon(in_size);
    CUDA_CHECK(cudaMemcpy(h_recon.data(), d_recon,
                          in_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "\nSample reconstructed values (first 10):\n";
    for (int i = 0; i < 10 && i < in_size; ++i) {
        std::cout << "  [" << i << "] input=" << h_input[i]
                  << ", recon=" << h_recon[i] << "\n";
    }

    CUDA_CHECK(cudaFree(d_features));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_recon));

    return 0;
}