#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <fstream>

#include "autoencoder_gpu.h"
#include "layers_gpu.h"

#define CUDA_CHECK(cmd)                                     \
    do {                                                    \
        cudaError_t e = (cmd);                              \
        if (e != cudaSuccess) {                             \
            printf("CUDA Error %s:%d: %s\n",                \
                   __FILE__, __LINE__,                      \
                   cudaGetErrorString(e));                  \
        }                                                   \
    } while (0)

AutoencoderGPU::AutoencoderGPU(int N, int H, int W, float lr)
    : N_(N),
      H_(H),
      W_(W),
      K_(3),
      lr_(lr),
      d_w1_(nullptr), d_b1_(nullptr),
      d_w2_(nullptr), d_b2_(nullptr),
      d_w3_(nullptr), d_b3_(nullptr),
      d_w4_(nullptr), d_b4_(nullptr),
      d_w5_(nullptr), d_b5_(nullptr),
      d_dw1_(nullptr), d_db1_(nullptr),
      d_dw2_(nullptr), d_db2_(nullptr),
      d_dw3_(nullptr), d_db3_(nullptr),
      d_dw4_(nullptr), d_db4_(nullptr),
      d_dw5_(nullptr), d_db5_(nullptr),
      d_conv1_(nullptr), d_relu1_(nullptr), d_pool1_(nullptr),
      d_conv2_(nullptr), d_relu2_(nullptr), d_pool2_(nullptr),
      d_conv3_(nullptr), d_relu3_(nullptr), d_up1_(nullptr),
      d_conv4_(nullptr), d_relu4_(nullptr), d_up2_(nullptr),
      d_conv5_(nullptr),
      d_dconv1_(nullptr), d_drelu1_(nullptr), d_dpool1_(nullptr),
      d_dconv2_(nullptr), d_drelu2_(nullptr), d_dpool2_(nullptr),
      d_dconv3_(nullptr), d_drelu3_(nullptr), d_dup1_(nullptr),
      d_dconv4_(nullptr), d_drelu4_(nullptr), d_dup2_(nullptr),
      d_dconv5_(nullptr),
      d_drecon_(nullptr) {

    // Tính kích thước các layer
    int H16 = H_ / 2;  // sau pool1
    int W16 = W_ / 2;
    int H8 = H_ / 4;   // sau pool2 (latent)
    int W8 = W_ / 4;

    // Allocate weights & biases (5 conv layers)
    // Conv1: 3→256
    CUDA_CHECK(cudaMalloc(&d_w1_, C1_ * C_in_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1_, C1_ * sizeof(float)));
    // Conv2: 256→128
    CUDA_CHECK(cudaMalloc(&d_w2_, C2_ * C1_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2_, C2_ * sizeof(float)));
    // Conv3: 128→128
    CUDA_CHECK(cudaMalloc(&d_w3_, C3_ * C2_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3_, C3_ * sizeof(float)));
    // Conv4: 128→256
    CUDA_CHECK(cudaMalloc(&d_w4_, C4_ * C3_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b4_, C4_ * sizeof(float)));
    // Conv5: 256→3
    CUDA_CHECK(cudaMalloc(&d_w5_, C5_ * C4_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b5_, C5_ * sizeof(float)));

    // Allocate gradients
    CUDA_CHECK(cudaMalloc(&d_dw1_, C1_ * C_in_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1_, C1_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2_, C2_ * C1_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2_, C2_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw3_, C3_ * C2_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db3_, C3_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw4_, C4_ * C3_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db4_, C4_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw5_, C5_ * C4_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db5_, C5_ * sizeof(float)));

    // Allocate activations
    // Encoder
    CUDA_CHECK(cudaMalloc(&d_conv1_, N_ * C1_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu1_, N_ * C1_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_, N_ * C1_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_, N_ * C2_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu2_, N_ * C2_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2_, N_ * C2_ * H8 * W8 * sizeof(float))); // LATENT
    // Decoder
    CUDA_CHECK(cudaMalloc(&d_conv3_, N_ * C3_ * H8 * W8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu3_, N_ * C3_ * H8 * W8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up1_, N_ * C3_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_, N_ * C4_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu4_, N_ * C4_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up2_, N_ * C4_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_, N_ * C5_ * H_ * W_ * sizeof(float))); // OUTPUT

    // Allocate activation gradients
    CUDA_CHECK(cudaMalloc(&d_dconv1_, N_ * C1_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_drelu1_, N_ * C1_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dpool1_, N_ * C1_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dconv2_, N_ * C2_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_drelu2_, N_ * C2_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dpool2_, N_ * C2_ * H8 * W8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dconv3_, N_ * C3_ * H8 * W8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_drelu3_, N_ * C3_ * H8 * W8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dup1_, N_ * C3_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dconv4_, N_ * C4_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_drelu4_, N_ * C4_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dup2_, N_ * C4_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dconv5_, N_ * C5_ * H_ * W_ * sizeof(float)));

    // Allocate reusable buffer for dL/d(recon) to avoid per-iteration malloc/free
    int total_output = N_ * C5_ * H_ * W_;
    CUDA_CHECK(cudaMalloc(&d_drecon_, total_output * sizeof(float)));

    // Khởi tạo weights ngẫu nhiên nhỏ
    std::vector<float> h_w1(C1_ * C_in_ * K_ * K_);
    std::vector<float> h_w2(C2_ * C1_ * K_ * K_);
    std::vector<float> h_w3(C3_ * C2_ * K_ * K_);
    std::vector<float> h_w4(C4_ * C3_ * K_ * K_);
    std::vector<float> h_w5(C5_ * C4_ * K_ * K_);
    std::vector<float> h_b1(C1_, 0.0f);
    std::vector<float> h_b2(C2_, 0.0f);
    std::vector<float> h_b3(C3_, 0.0f);
    std::vector<float> h_b4(C4_, 0.0f);
    std::vector<float> h_b5(C5_, 0.0f);

    // Init uniform nhỏ [-0.1, 0.1]
    for (size_t i = 0; i < h_w1.size(); ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        h_w1[i] = (r * 0.2f - 0.1f);
    }
    for (size_t i = 0; i < h_w2.size(); ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        h_w2[i] = (r * 0.2f - 0.1f);
    }
    for (size_t i = 0; i < h_w3.size(); ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        h_w3[i] = (r * 0.2f - 0.1f);
    }
    for (size_t i = 0; i < h_w4.size(); ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        h_w4[i] = (r * 0.2f - 0.1f);
    }
    for (size_t i = 0; i < h_w5.size(); ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        h_w5[i] = (r * 0.2f - 0.1f);
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_w1_, h_w1.data(), h_w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2_, h_w2.data(), h_w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3_, h_w3.data(), h_w3.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w4_, h_w4.data(), h_w4.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w5_, h_w5.data(), h_w5.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1_, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2_, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3_, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b4_, h_b4.data(), h_b4.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b5_, h_b5.data(), h_b5.size() * sizeof(float), cudaMemcpyHostToDevice));
}

AutoencoderGPU::~AutoencoderGPU() {
    // Free weights
    CUDA_CHECK(cudaFree(d_w1_)); CUDA_CHECK(cudaFree(d_b1_));
    CUDA_CHECK(cudaFree(d_w2_)); CUDA_CHECK(cudaFree(d_b2_));
    CUDA_CHECK(cudaFree(d_w3_)); CUDA_CHECK(cudaFree(d_b3_));
    CUDA_CHECK(cudaFree(d_w4_)); CUDA_CHECK(cudaFree(d_b4_));
    CUDA_CHECK(cudaFree(d_w5_)); CUDA_CHECK(cudaFree(d_b5_));

    // Free gradients
    CUDA_CHECK(cudaFree(d_dw1_)); CUDA_CHECK(cudaFree(d_db1_));
    CUDA_CHECK(cudaFree(d_dw2_)); CUDA_CHECK(cudaFree(d_db2_));
    CUDA_CHECK(cudaFree(d_dw3_)); CUDA_CHECK(cudaFree(d_db3_));
    CUDA_CHECK(cudaFree(d_dw4_)); CUDA_CHECK(cudaFree(d_db4_));
    CUDA_CHECK(cudaFree(d_dw5_)); CUDA_CHECK(cudaFree(d_db5_));

    // Free activations
    CUDA_CHECK(cudaFree(d_conv1_)); CUDA_CHECK(cudaFree(d_relu1_)); CUDA_CHECK(cudaFree(d_pool1_));
    CUDA_CHECK(cudaFree(d_conv2_)); CUDA_CHECK(cudaFree(d_relu2_)); CUDA_CHECK(cudaFree(d_pool2_));
    CUDA_CHECK(cudaFree(d_conv3_)); CUDA_CHECK(cudaFree(d_relu3_)); CUDA_CHECK(cudaFree(d_up1_));
    CUDA_CHECK(cudaFree(d_conv4_)); CUDA_CHECK(cudaFree(d_relu4_)); CUDA_CHECK(cudaFree(d_up2_));
    CUDA_CHECK(cudaFree(d_conv5_));

    // Free activation gradients
    CUDA_CHECK(cudaFree(d_dconv1_)); CUDA_CHECK(cudaFree(d_drelu1_)); CUDA_CHECK(cudaFree(d_dpool1_));
    CUDA_CHECK(cudaFree(d_dconv2_)); CUDA_CHECK(cudaFree(d_drelu2_)); CUDA_CHECK(cudaFree(d_dpool2_));
    CUDA_CHECK(cudaFree(d_dconv3_)); CUDA_CHECK(cudaFree(d_drelu3_)); CUDA_CHECK(cudaFree(d_dup1_));
    CUDA_CHECK(cudaFree(d_dconv4_)); CUDA_CHECK(cudaFree(d_drelu4_)); CUDA_CHECK(cudaFree(d_dup2_));
    CUDA_CHECK(cudaFree(d_dconv5_));
    CUDA_CHECK(cudaFree(d_drecon_));
}

void AutoencoderGPU::forward(const float* d_input, float* d_recon) {
    int H16 = H_ / 2;
    int W16 = W_ / 2;
    int H8 = H_ / 4;
    int W8 = W_ / 4;

    // ENCODER
    // Conv1: 3→256, ReLU
    conv2d_forward_gpu_naive(
        d_input, d_w1_, d_b1_, d_conv1_,
        N_, C_in_, H_, W_, C1_, K_);
    relu_forward_gpu(d_conv1_, d_relu1_, N_ * C1_ * H_ * W_);

    // MaxPool1: (32,32) → (16,16)
    maxpool2d_forward_gpu(d_relu1_, d_pool1_, N_, C1_, H_, W_);

    // Conv2: 256→128, ReLU
    conv2d_forward_gpu_naive(
        d_pool1_, d_w2_, d_b2_, d_conv2_,
        N_, C1_, H16, W16, C2_, K_);
    relu_forward_gpu(d_conv2_, d_relu2_, N_ * C2_ * H16 * W16);

    // MaxPool2: (16,16) → (8,8) = LATENT
    maxpool2d_forward_gpu(d_relu2_, d_pool2_, N_, C2_, H16, W16);

    // DECODER
    // Conv3: 128→128, ReLU
    conv2d_forward_gpu_naive(
        d_pool2_, d_w3_, d_b3_, d_conv3_,
        N_, C2_, H8, W8, C3_, K_);
    relu_forward_gpu(d_conv3_, d_relu3_, N_ * C3_ * H8 * W8);

    // Upsample1: (8,8) → (16,16)
    upsample2d_forward_gpu(d_relu3_, d_up1_, N_, C3_, H8, W8);

    // Conv4: 128→256, ReLU
    conv2d_forward_gpu_naive(
        d_up1_, d_w4_, d_b4_, d_conv4_,
        N_, C3_, H16, W16, C4_, K_);
    relu_forward_gpu(d_conv4_, d_relu4_, N_ * C4_ * H16 * W16);

    // Upsample2: (16,16) → (32,32)
    upsample2d_forward_gpu(d_relu4_, d_up2_, N_, C4_, H16, W16);

    // Conv5: 256→3 (no activation)
    conv2d_forward_gpu_naive(
        d_up2_, d_w5_, d_b5_, d_recon,
        N_, C4_, H_, W_, C5_, K_);
}

void AutoencoderGPU::extract_features(const float* d_input, float* d_features) {
    int H16 = H_ / 2;
    int W16 = W_ / 2;
    int H8 = H_ / 4;
    int W8 = W_ / 4;

    // Chỉ chạy encoder
    // Conv1: 3→256, ReLU
    conv2d_forward_gpu_naive(
        d_input, d_w1_, d_b1_, d_conv1_,
        N_, C_in_, H_, W_, C1_, K_);
    relu_forward_gpu(d_conv1_, d_relu1_, N_ * C1_ * H_ * W_);

    // MaxPool1
    maxpool2d_forward_gpu(d_relu1_, d_pool1_, N_, C1_, H_, W_);

    // Conv2: 256→128, ReLU
    conv2d_forward_gpu_naive(
        d_pool1_, d_w2_, d_b2_, d_conv2_,
        N_, C1_, H16, W16, C2_, K_);
    relu_forward_gpu(d_conv2_, d_relu2_, N_ * C2_ * H16 * W16);

    // MaxPool2: (8,8,128) = LATENT
    maxpool2d_forward_gpu(d_relu2_, d_pool2_, N_, C2_, H16, W16);

    // Copy latent to output
    int latent_size = N_ * C2_ * H8 * W8;
    CUDA_CHECK(cudaMemcpy(d_features, d_pool2_, latent_size * sizeof(float), cudaMemcpyDeviceToDevice));
}

float AutoencoderGPU::train_step(const float* d_input, float* d_recon) {
    // Forward
    forward(d_input, d_recon);

    int total = N_ * C5_ * H_ * W_;

    // Tính loss
    float loss = mse_loss_forward_gpu(d_recon, d_input, total);

    // dL/dPred (re-use preallocated buffer to avoid per-iteration malloc/free)
    mse_loss_backward_gpu(d_recon, d_input, d_drecon_, total);

    // Backward
    backward(d_input, d_recon, d_drecon_);

    // Update
    step();

    return loss;
}

void AutoencoderGPU::backward(const float* d_input,
                              const float* d_recon,
                              const float* d_drecon) {
    int H16 = H_ / 2;
    int W16 = W_ / 2;
    int H8 = H_ / 4;
    int W8 = W_ / 4;

    // Zero gradients
    CUDA_CHECK(cudaMemset(d_dw1_, 0, C1_ * C_in_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db1_, 0, C1_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dw2_, 0, C2_ * C1_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db2_, 0, C2_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dw3_, 0, C3_ * C2_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db3_, 0, C3_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dw4_, 0, C4_ * C3_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db4_, 0, C4_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dw5_, 0, C5_ * C4_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db5_, 0, C5_ * sizeof(float)));

    // DECODER BACKWARD
    // 1) Conv5 backward: d_drecon -> d_dup2_
    conv2d_backward_gpu_naive(
        d_drecon, d_up2_, d_w5_,
        d_dup2_, d_dw5_, d_db5_,
        N_, C4_, H_, W_, C5_, K_);

    // 2) Upsample2 backward: d_dup2_ -> d_drelu4_
    upsample2d_backward_gpu(d_dup2_, d_drelu4_, N_, C4_, H16, W16);

    // 3) ReLU4 backward: d_drelu4_ -> d_dconv4_ (đúng thứ tự!)
    relu_backward_gpu(d_drelu4_, d_relu4_, d_dconv4_, N_ * C4_ * H16 * W16);

    // 4) Conv4 backward: d_dconv4_ -> d_dup1_
    conv2d_backward_gpu_naive(
        d_dconv4_, d_up1_, d_w4_,
        d_dup1_, d_dw4_, d_db4_,
        N_, C3_, H16, W16, C4_, K_);

    // 5) Upsample1 backward: d_dup1_ -> d_drelu3_
    upsample2d_backward_gpu(d_dup1_, d_drelu3_, N_, C3_, H8, W8);

    // 6) ReLU3 backward: d_drelu3_ -> d_dconv3_ (đúng thứ tự!)
    relu_backward_gpu(d_drelu3_, d_relu3_, d_dconv3_, N_ * C3_ * H8 * W8);

    // 7) Conv3 backward: d_dconv3_ -> d_dpool2_
    conv2d_backward_gpu_naive(
        d_dconv3_, d_pool2_, d_w3_,
        d_dpool2_, d_dw3_, d_db3_,
        N_, C2_, H8, W8, C3_, K_);

    // ENCODER BACKWARD
    // 8) MaxPool2 backward: d_dpool2_ -> d_drelu2_
    maxpool2d_backward_gpu(d_dpool2_, d_relu2_, d_drelu2_, N_, C2_, H16, W16);

    // 9) ReLU2 backward: d_drelu2_ -> d_dconv2_ (đúng thứ tự!)
    relu_backward_gpu(d_drelu2_, d_relu2_, d_dconv2_, N_ * C2_ * H16 * W16);

    // 10) Conv2 backward: d_dconv2_ -> d_dpool1_
    conv2d_backward_gpu_naive(
        d_dconv2_, d_pool1_, d_w2_,
        d_dpool1_, d_dw2_, d_db2_,
        N_, C1_, H16, W16, C2_, K_);

    // 11) MaxPool1 backward: d_dpool1_ -> d_drelu1_
    maxpool2d_backward_gpu(d_dpool1_, d_relu1_, d_drelu1_, N_, C1_, H_, W_);

    // 12) ReLU1 backward: d_drelu1_ -> d_dconv1_ (đúng thứ tự!)
    relu_backward_gpu(d_drelu1_, d_relu1_, d_dconv1_, N_ * C1_ * H_ * W_);

    // 13) Conv1 backward: d_dconv1_ -> d_dinput (không cần output nhưng cần buffer hợp lệ)
    conv2d_backward_gpu_naive(
        d_dconv1_, d_input, d_w1_,
        d_dconv1_, d_dw1_, d_db1_,  // d_dinput dùng lại buffer d_dconv1_ (không cần output)
        N_, C_in_, H_, W_, C1_, K_);
}

// Kernel SGD update: param -= lr * grad
__global__ void sgd_update_kernel(float* param,
                                  const float* grad,
                                  float lr,
                                  int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    param[idx] -= lr * grad[idx];
}

void AutoencoderGPU::step() {
    int block = 256;

    // Update all 5 conv layers
    int n_w1 = C1_ * C_in_ * K_ * K_;
    int n_w2 = C2_ * C1_ * K_ * K_;
    int n_w3 = C3_ * C2_ * K_ * K_;
    int n_w4 = C4_ * C3_ * K_ * K_;
    int n_w5 = C5_ * C4_ * K_ * K_;

    int grid_w1 = (n_w1 + block - 1) / block;
    int grid_w2 = (n_w2 + block - 1) / block;
    int grid_w3 = (n_w3 + block - 1) / block;
    int grid_w4 = (n_w4 + block - 1) / block;
    int grid_w5 = (n_w5 + block - 1) / block;

    int grid_b1 = (C1_ + block - 1) / block;
    int grid_b2 = (C2_ + block - 1) / block;
    int grid_b3 = (C3_ + block - 1) / block;
    int grid_b4 = (C4_ + block - 1) / block;
    int grid_b5 = (C5_ + block - 1) / block;

    sgd_update_kernel<<<grid_w1, block>>>(d_w1_, d_dw1_, lr_, n_w1);
    sgd_update_kernel<<<grid_b1, block>>>(d_b1_, d_db1_, lr_, C1_);
    sgd_update_kernel<<<grid_w2, block>>>(d_w2_, d_dw2_, lr_, n_w2);
    sgd_update_kernel<<<grid_b2, block>>>(d_b2_, d_db2_, lr_, C2_);
    sgd_update_kernel<<<grid_w3, block>>>(d_w3_, d_dw3_, lr_, n_w3);
    sgd_update_kernel<<<grid_b3, block>>>(d_b3_, d_db3_, lr_, C3_);
    sgd_update_kernel<<<grid_w4, block>>>(d_w4_, d_dw4_, lr_, n_w4);
    sgd_update_kernel<<<grid_b4, block>>>(d_b4_, d_db4_, lr_, C4_);
    sgd_update_kernel<<<grid_w5, block>>>(d_w5_, d_dw5_, lr_, n_w5);
    sgd_update_kernel<<<grid_b5, block>>>(d_b5_, d_db5_, lr_, C5_);

    CUDA_CHECK(cudaGetLastError());
}

void AutoencoderGPU::save_weights(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        printf("ERROR: Cannot open file for writing: %s\n", filepath.c_str());
        return;
    }

    // Tính kích thước mỗi layer
    int n_w1 = C1_ * C_in_ * K_ * K_;
    int n_w2 = C2_ * C1_ * K_ * K_;
    int n_w3 = C3_ * C2_ * K_ * K_;
    int n_w4 = C4_ * C3_ * K_ * K_;
    int n_w5 = C5_ * C4_ * K_ * K_;

    // Copy weights từ GPU về host
    std::vector<float> h_w1(n_w1), h_w2(n_w2), h_w3(n_w3), h_w4(n_w4), h_w5(n_w5);
    std::vector<float> h_b1(C1_), h_b2(C2_), h_b3(C3_), h_b4(C4_), h_b5(C5_);

    CUDA_CHECK(cudaMemcpy(h_w1.data(), d_w1_, n_w1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1.data(), d_b1_, C1_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w2.data(), d_w2_, n_w2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2.data(), d_b2_, C2_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w3.data(), d_w3_, n_w3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b3.data(), d_b3_, C3_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w4.data(), d_w4_, n_w4 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b4.data(), d_b4_, C4_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w5.data(), d_w5_, n_w5 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b5.data(), d_b5_, C5_ * sizeof(float), cudaMemcpyDeviceToHost));

    // Ghi vào file: format đơn giản
    // Header: số layers (5)
    int num_layers = 5;
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));

    // Layer 1: Conv1 (3→256)
    file.write(reinterpret_cast<const char*>(&n_w1), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w1.data()), n_w1 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b1.data()), C1_ * sizeof(float));

    // Layer 2: Conv2 (256→128)
    file.write(reinterpret_cast<const char*>(&n_w2), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w2.data()), n_w2 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b2.data()), C2_ * sizeof(float));

    // Layer 3: Conv3 (128→128)
    file.write(reinterpret_cast<const char*>(&n_w3), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w3.data()), n_w3 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b3.data()), C3_ * sizeof(float));

    // Layer 4: Conv4 (128→256)
    file.write(reinterpret_cast<const char*>(&n_w4), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w4.data()), n_w4 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b4.data()), C4_ * sizeof(float));

    // Layer 5: Conv5 (256→3)
    file.write(reinterpret_cast<const char*>(&n_w5), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w5.data()), n_w5 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b5.data()), C5_ * sizeof(float));

    file.close();
    printf("✓ Saved weights to %s\n", filepath.c_str());
}

void AutoencoderGPU::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        printf("ERROR: Cannot open file for reading: %s\n", filepath.c_str());
        return;
    }

    // Đọc header
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    if (num_layers != 5) {
        printf("ERROR: File has %d layers, expected 5\n", num_layers);
        file.close();
        return;
    }

    // Tính kích thước mỗi layer
    int n_w1 = C1_ * C_in_ * K_ * K_;
    int n_w2 = C2_ * C1_ * K_ * K_;
    int n_w3 = C3_ * C2_ * K_ * K_;
    int n_w4 = C4_ * C3_ * K_ * K_;
    int n_w5 = C5_ * C4_ * K_ * K_;

    // Đọc và load từng layer
    std::vector<float> h_w1(n_w1), h_w2(n_w2), h_w3(n_w3), h_w4(n_w4), h_w5(n_w5);
    std::vector<float> h_b1(C1_), h_b2(C2_), h_b3(C3_), h_b4(C4_), h_b5(C5_);

    int n_read;
    
    // Layer 1
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    if (n_read != n_w1) {
        printf("ERROR: Layer 1 size mismatch\n");
        file.close();
        return;
    }
    file.read(reinterpret_cast<char*>(h_w1.data()), n_w1 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b1.data()), C1_ * sizeof(float));

    // Layer 2
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    file.read(reinterpret_cast<char*>(h_w2.data()), n_w2 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b2.data()), C2_ * sizeof(float));

    // Layer 3
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    file.read(reinterpret_cast<char*>(h_w3.data()), n_w3 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b3.data()), C3_ * sizeof(float));

    // Layer 4
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    file.read(reinterpret_cast<char*>(h_w4.data()), n_w4 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b4.data()), C4_ * sizeof(float));

    // Layer 5
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    file.read(reinterpret_cast<char*>(h_w5.data()), n_w5 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b5.data()), C5_ * sizeof(float));

    file.close();

    // Copy lên GPU
    CUDA_CHECK(cudaMemcpy(d_w1_, h_w1.data(), n_w1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1_, h_b1.data(), C1_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2_, h_w2.data(), n_w2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2_, h_b2.data(), C2_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3_, h_w3.data(), n_w3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3_, h_b3.data(), C3_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w4_, h_w4.data(), n_w4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b4_, h_b4.data(), C4_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w5_, h_w5.data(), n_w5 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b5_, h_b5.data(), C5_ * sizeof(float), cudaMemcpyHostToDevice));

    printf("✓ Loaded weights from %s\n", filepath.c_str());
}
