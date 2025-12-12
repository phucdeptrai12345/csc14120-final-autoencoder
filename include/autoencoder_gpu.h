#ifndef AUTOENCODER_GPU_H
#define AUTOENCODER_GPU_H

#include <cuda_runtime.h>
#include <string>

// Sử dụng các kernel/layer đã định nghĩa trong layers_gpu.h
#include "layers_gpu.h"

// Autoencoder theo spec project:
// ENCODER:
//   INPUT: (32, 32, 3)
//   Conv1: 3→256, ReLU, MaxPool(2x2) → (16, 16, 256)
//   Conv2: 256→128, ReLU, MaxPool(2x2) → (8, 8, 128) = LATENT
// DECODER:
//   Conv3: 128→128, ReLU, Upsample(2x) → (16, 16, 128)
//   Conv4: 128→256, ReLU, Upsample(2x) → (32, 32, 256)
//   Conv5: 256→3 (no activation) → (32, 32, 3)
//
// Tổng cộng 5 conv layers với kernel 3x3, padding=1

class AutoencoderGPU {
public:
    // Constructor: nhận batch size N, input size HxW (thường 32x32 cho CIFAR-10)
    // Architecture cố định: 3→256→128→128→256→3
    AutoencoderGPU(int N, int H, int W, float lr = 1e-3f);

    ~AutoencoderGPU();

    // Không cho copy để tránh double-free
    AutoencoderGPU(const AutoencoderGPU&) = delete;
    AutoencoderGPU& operator=(const AutoencoderGPU&) = delete;

    // Forward: input (device pointer, shape N x 3 x H x W)
    // Output: recon (device pointer, shape N x 3 x H x W)
    // actual_N (optional): số mẫu thực tế trong buffer. Nếu <=0 dùng N_ (đã khởi tạo).
    void forward(const float* d_input, float* d_recon, int actual_N = -1);

    // Feature extraction: chỉ chạy encoder, trả về latent (N x 128 x 8 x 8)
    // Output: d_features (device pointer, shape actual_N x 128 x 8 x 8)
    void extract_features(const float* d_input, float* d_features, int actual_N = -1);

    // Backward toàn bộ AE với MSE loss:
    // - Trả về loss trên host (float)
    // actual_N (optional): số mẫu thực tế — dùng để tính total elements / gradient scale
    float train_step(const float* d_input, float* d_recon, int actual_N = -1);

    // Chạy backward (sau khi đã có d_recon từ MSE loss)
    void backward(const float* d_input,
                  const float* d_recon,
                  const float* d_drecon,
                  int actual_N = -1);

    // SGD update: w -= lr * dw, b -= lr * db
    void step();

    // Save/Load weights
    // Lưu tất cả weights và biases của 5 conv layers ra file
    void save_weights(const std::string& filepath) const;
    
    // Load weights từ file (phải khớp architecture)
    void load_weights(const std::string& filepath);

    // Truy cập thông tin
    int batch_size() const { return N_; }
    int height() const { return H_; }
    int width() const { return W_; }
    int latent_size() const { return N_ * 128 * 8 * 8; } // (8,8,128) = 8192 per sample

private:
    int N_, H_, W_;
    int K_; // kernel size = 3
    float lr_;

    // Architecture channels (cố định theo spec)
    static const int C_in_ = 3;
    static const int C1_ = 256;  // Conv1: 3→256
    static const int C2_ = 128;  // Conv2: 256→128 (latent)
    static const int C3_ = 128;  // Conv3: 128→128
    static const int C4_ = 256;  // Conv4: 128→256
    static const int C5_ = 3;    // Conv5: 256→3

    // Weights & bias (device) - 5 conv layers
    float *d_w1_, *d_b1_;  // Conv1: 3→256
    float *d_w2_, *d_b2_;  // Conv2: 256→128
    float *d_w3_, *d_b3_;  // Conv3: 128→128
    float *d_w4_, *d_b4_;  // Conv4: 128→256
    float *d_w5_, *d_b5_;   // Conv5: 256→3

    // Gradients (device)
    float *d_dw1_, *d_db1_;
    float *d_dw2_, *d_db2_;
    float *d_dw3_, *d_db3_;
    float *d_dw4_, *d_db4_;
    float *d_dw5_, *d_db5_;

    // Intermediate activations (device)
    // Encoder path:
    float *d_conv1_, *d_relu1_;      // (N, 256, 32, 32)
    float *d_pool1_;                 // (N, 256, 16, 16)
    float *d_conv2_, *d_relu2_;      // (N, 128, 16, 16)
    float *d_pool2_;                 // (N, 128, 8, 8) = LATENT
    // Decoder path:
    float *d_conv3_, *d_relu3_;      // (N, 128, 8, 8)
    float *d_up1_;                   // (N, 128, 16, 16)
    float *d_conv4_, *d_relu4_;      // (N, 256, 16, 16)
    float *d_up2_;                   // (N, 256, 32, 32)
    float *d_conv5_;                 // (N, 3, 32, 32) = OUTPUT

    // Gradients w.r.t activations (device)
    float *d_dconv1_, *d_drelu1_, *d_dpool1_;
    float *d_dconv2_, *d_drelu2_, *d_dpool2_;
    float *d_dconv3_, *d_drelu3_, *d_dup1_;
    float *d_dconv4_, *d_drelu4_, *d_dup2_;
    float *d_dconv5_;
    float *d_dinput1_; // gradient wrt original input of conv1 (separate buffer)
    // Reusable buffer for dL/d(recon)
    float *d_drecon_;
};

#endif // AUTOENCODER_GPU_H