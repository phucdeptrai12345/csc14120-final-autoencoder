#ifndef AUTOENCODER_GPU_OPTIMIZED_H
#define AUTOENCODER_GPU_OPTIMIZED_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For __half type (FP16 support)
#include <string>

// Phase 3: Optimized kernels
#include "layers_gpu_optimized.h"
// Also need naive kernels for backward and Conv5 (no ReLU)
#include "layers_gpu.h"

// Autoencoder theo spec project (same as Phase 2):
// ENCODER:
//   INPUT: (32, 32, 3)
//   Conv1: 3→256, ReLU, MaxPool(2x2) → (16, 16, 256)
//   Conv2: 256→128, ReLU, MaxPool(2x2) → (8, 8, 128) = LATENT
// DECODER:
//   Conv3: 128→128, ReLU, Upsample(2x) → (16, 16, 128)
//   Conv4: 128→256, ReLU, Upsample(2x) → (32, 32, 256)
//   Conv5: 256→3 (no activation) → (32, 32, 3)
//
// OPTIMIZATIONS:
// - Step 1: Kernel Fusion (Conv+ReLU fused) for Conv1-4
// - Step 2: Keep d_conv buffers for backward correctness
// - Step 3: Pinned memory in training loop

class AutoencoderGPUOptimized {
public:
    AutoencoderGPUOptimized(int N, int H, int W, float lr = 1e-3f);
    ~AutoencoderGPUOptimized();

    AutoencoderGPUOptimized(const AutoencoderGPUOptimized&) = delete;
    AutoencoderGPUOptimized& operator=(const AutoencoderGPUOptimized&) = delete;

    // Forward: uses FUSED Conv+ReLU kernels (Step 1 optimization)
    void forward(const float* d_input, float* d_recon, cudaStream_t stream = 0);

    // Feature extraction: encoder only
    void extract_features(const float* d_input, float* d_features, cudaStream_t stream = 0);

    // Training step: forward + backward + update
    float train_step(const float* d_input, float* d_recon,
                     cudaStream_t stream = 0,
                     bool compute_loss_host = false,
                     float* h_loss_out = nullptr);

    // Backward pass (uses naive kernels for now)
    void backward(const float* d_input,
                  const float* d_recon,
                  const float* d_drecon,
                  cudaStream_t stream = 0);

    // SGD update
    void step(cudaStream_t stream = 0);

    // Save/Load weights
    void save_weights(const std::string& filepath) const;
    void load_weights(const std::string& filepath);

    // Accessors
    int batch_size() const { return N_; }
    int height() const { return H_; }
    int width() const { return W_; }
    int latent_size() const { return N_ * 128 * 8 * 8; }

private:
    int N_, H_, W_;
    int K_; // kernel size = 3
    float lr_;

    // Architecture channels (same as Phase 2)
    static const int C_in_ = 3;
    static const int C1_ = 256;  // Conv1: 3→256
    static const int C2_ = 128;  // Conv2: 256→128 (latent)
    static const int C3_ = 128;  // Conv3: 128→128
    static const int C4_ = 256;  // Conv4: 128→256
    static const int C5_ = 3;    // Conv5: 256→3

    // Weights & bias (device) - 5 conv layers
    float *d_w1_, *d_b1_;
    float *d_w2_, *d_b2_;
    float *d_w3_, *d_b3_;
    float *d_w4_, *d_b4_;
    float *d_w5_, *d_b5_;

    // Gradients (device)
    float *d_dw1_, *d_db1_;
    float *d_dw2_, *d_db2_;
    float *d_dw3_, *d_db3_;
    float *d_dw4_, *d_db4_;
    float *d_dw5_, *d_db5_;

    // Intermediate activations (device)
    // Step 2: Keep d_conv buffers for backward correctness
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
    // Reusable buffer for dL/d(recon)
    float *d_drecon_;
    // Temporary buffer for Conv1 backward input gradient (tránh race condition)
    float *d_dinput_temp_;
    
    // Mixed Precision (FP16) buffers - reused across forward passes
    bool use_mixed_precision_;
    bool gpu_supports_fp16_;
    // FP16 buffers for activations (reused to avoid allocation overhead)
    // Need separate input/output buffers to avoid race conditions
    __half *d_fp16_input1_, *d_fp16_output1_;  // For Conv1 (input: N*3*32*32, output: N*256*32*32)
    __half *d_fp16_input2_, *d_fp16_output2_;  // For Conv2 (input: N*256*16*16, output: N*128*16*16)
    __half *d_fp16_input3_, *d_fp16_output3_;  // For Conv3 (input: N*128*8*8, output: N*128*8*8)
    __half *d_fp16_input4_, *d_fp16_output4_;  // For Conv4 (input: N*128*16*16, output: N*256*16*16)
    
    // Helper: Check GPU FP16 support
    bool check_fp16_support();
};

#endif // AUTOENCODER_GPU_OPTIMIZED_H

