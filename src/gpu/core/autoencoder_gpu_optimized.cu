// ============================================
// autoencoder_gpu_optimized.cu - PHASE 3 FINAL
// Complete implementation with optimizations + fixes
// ============================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For FP16 support (must be before layers_gpu_optimized.h)
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include "autoencoder_gpu_optimized.h"
#include "layers_gpu_optimized.h"
#include "layers_gpu.h"

// Forward declarations for optimized kernels (defined in layers_gpu_optimized.cu)
extern void maxpool2d_forward_gpu_optimized(
    const float* d_input, float* d_output,
    int N, int C, int H, int W, cudaStream_t stream);
extern void upsample2d_forward_gpu_optimized(
    const float* d_input, float* d_output,
    int N, int C, int H, int W, cudaStream_t stream);

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = (cmd); \
        if (e != cudaSuccess) { \
            printf("CUDA Error %s:%d: %s\n", \
                   __FILE__, __LINE__, \
                   cudaGetErrorString(e)); \
            exit(1); \
        } \
    } while (0)

// Helper: Glorot initialization
void glorot_init(std::vector<float>& w, int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    for (size_t i = 0; i < w.size(); ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        w[i] = r * 2.0f * limit - limit;
    }
}

AutoencoderGPUOptimized::AutoencoderGPUOptimized(int N, int H, int W, float lr)
    : N_(N), H_(H), W_(W), K_(3), lr_(lr),
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
      d_drecon_(nullptr),
      d_dinput_temp_(nullptr),
      d_im2col_(nullptr),
      d_gemm_out_(nullptr),
      d_im2col_fp16_(nullptr),
      cublas_handle_(nullptr),
      use_mixed_precision_(false),
      gpu_supports_fp16_(false)
{
    int H16 = H_ / 2;
    int W16 = W_ / 2;
    int H8 = H_ / 4;
    int W8 = W_ / 4;

    // === ALLOCATE WEIGHTS & BIASES ===
    CUDA_CHECK(cudaMalloc(&d_w1_, C1_ * C_in_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1_, C1_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2_, C2_ * C1_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2_, C2_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3_, C3_ * C2_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3_, C3_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w4_, C4_ * C3_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b4_, C4_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w5_, C5_ * C4_ * K_ * K_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b5_, C5_ * sizeof(float)));

    // === ALLOCATE GRADIENTS ===
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

    // === ALLOCATE ACTIVATIONS ===
    CUDA_CHECK(cudaMalloc(&d_conv1_, N_ * C1_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu1_, N_ * C1_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_, N_ * C1_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_, N_ * C2_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu2_, N_ * C2_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2_, N_ * C2_ * H8 * W8 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_conv3_, N_ * C3_ * H8 * W8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu3_, N_ * C3_ * H8 * W8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up1_, N_ * C3_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_, N_ * C4_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu4_, N_ * C4_ * H16 * W16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up2_, N_ * C4_ * H_ * W_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_, N_ * C5_ * H_ * W_ * sizeof(float)));

    // === ALLOCATE ACTIVATION GRADIENTS ===
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
    
    // === REUSABLE BUFFERS ===
    int total_output = N_ * C5_ * H_ * W_;
    CUDA_CHECK(cudaMalloc(&d_drecon_, total_output * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dinput_temp_, N_ * C_in_ * H_ * W_ * sizeof(float)));

    // === ALLOCATE GEMM WORKSPACES ===
    // im2col max (Conv2 dominates): N * C1 * K*K * H16 * W16
    size_t im2col_max = static_cast<size_t>(N_) * C1_ * K_ * K_ * (H16 * W16);
    size_t im2col_bytes = im2col_max * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_im2col_, im2col_bytes));
    // GEMM out max across Conv1-4
    size_t out_max = static_cast<size_t>(N_) * std::max({
        C1_ * H_ * W_,
        C2_ * H16 * W16,
        C3_ * H8 * W8,
        C4_ * H16 * W16
    });
    size_t out_bytes = out_max * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_gemm_out_, out_bytes));

    // === cuBLAS handle ===
    cublasStatus_t cstat = cublasCreate(&cublas_handle_);
    if (cstat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Error: cublasCreate failed (%d)\n", cstat);
        cublas_handle_ = nullptr;
        // We keep going — GEMM calls will likely fail; user should check logs.
    }

    // === INITIALIZE WEIGHTS (Glorot) ===
    int k2 = K_ * K_;
    std::vector<float> h_w1(C1_ * C_in_ * k2);
    std::vector<float> h_w2(C2_ * C1_ * k2);
    std::vector<float> h_w3(C3_ * C2_ * k2);
    std::vector<float> h_w4(C4_ * C3_ * k2);
    std::vector<float> h_w5(C5_ * C4_ * k2);
    std::vector<float> h_b1(C1_, 0.0f);
    std::vector<float> h_b2(C2_, 0.0f);
    std::vector<float> h_b3(C3_, 0.0f);
    std::vector<float> h_b4(C4_, 0.0f);
    std::vector<float> h_b5(C5_, 0.0f);
    glorot_init(h_w1, C_in_ * k2, C1_ * k2);
    glorot_init(h_w2, C1_ * k2, C2_ * k2);
    glorot_init(h_w3, C2_ * k2, C3_ * k2);
    glorot_init(h_w4, C3_ * k2, C4_ * k2);
    glorot_init(h_w5, C4_ * k2, C5_ * k2);
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
    
    // === CHECK GPU FP16 SUPPORT ===
    gpu_supports_fp16_ = check_fp16_support();
    use_mixed_precision_ = gpu_supports_fp16_;  // Auto-enable if GPU supports
    // Allow runtime override for benchmarking:
    //   set env DISABLE_FP16=1 to force FP32
    //   set env FORCE_FP16=1 to force FP16 (if supported)
    if (std::getenv("DISABLE_FP16")) {
        use_mixed_precision_ = false;
    }
    if (std::getenv("FORCE_FP16") && gpu_supports_fp16_) {
        use_mixed_precision_ = true;
    }
    
    // === ALLOCATE FP16 BUFFERS (if mixed precision enabled) ===
    if (use_mixed_precision_ && d_im2col_) {
        // Allocate FP16 im2col buffer for GEMM FP16 (Conv2-4)
        CUDA_CHECK(cudaMalloc(&d_im2col_fp16_, im2col_max * sizeof(__half)));
        printf("✓ Mixed Precision (FP16/FP32) enabled - GPU supports FP16\n");
        printf("  Strategy: Conv1 dùng FP32 (spatial nhỏ), Conv2-4 dùng FP16 GEMM (Tensor Cores)\n");
    } else {
        printf("⚠ Mixed Precision disabled - GPU does not support FP16 (compute capability < 7.0) or workspace allocation failed\n");
    }
}

AutoencoderGPUOptimized::~AutoencoderGPUOptimized() {
    cudaFree(d_w1_); cudaFree(d_b1_);
    cudaFree(d_w2_); cudaFree(d_b2_);
    cudaFree(d_w3_); cudaFree(d_b3_);
    cudaFree(d_w4_); cudaFree(d_b4_);
    cudaFree(d_w5_); cudaFree(d_b5_);
    cudaFree(d_dw1_); cudaFree(d_db1_);
    cudaFree(d_dw2_); cudaFree(d_db2_);
    cudaFree(d_dw3_); cudaFree(d_db3_);
    cudaFree(d_dw4_); cudaFree(d_db4_);
    cudaFree(d_dw5_); cudaFree(d_db5_);
    cudaFree(d_conv1_); cudaFree(d_relu1_); cudaFree(d_pool1_);
    cudaFree(d_conv2_); cudaFree(d_relu2_); cudaFree(d_pool2_);
    cudaFree(d_conv3_); cudaFree(d_relu3_); cudaFree(d_up1_);
    cudaFree(d_conv4_); cudaFree(d_relu4_); cudaFree(d_up2_);
    cudaFree(d_conv5_);
    cudaFree(d_dconv1_); cudaFree(d_drelu1_); cudaFree(d_dpool1_);
    cudaFree(d_dconv2_); cudaFree(d_drelu2_); cudaFree(d_dpool2_);
    cudaFree(d_dconv3_); cudaFree(d_drelu3_); cudaFree(d_dup1_);
    cudaFree(d_dconv4_); cudaFree(d_drelu4_); cudaFree(d_dup2_);
    cudaFree(d_dconv5_);
    cudaFree(d_drecon_);
    cudaFree(d_dinput_temp_);
    
    if (d_im2col_) cudaFree(d_im2col_);
    if (d_im2col_fp16_) cudaFree(d_im2col_fp16_);
    if (d_gemm_out_) cudaFree(d_gemm_out_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

// ============================================
// GPU FP16 SUPPORT CHECK
// ============================================
bool AutoencoderGPUOptimized::check_fp16_support() {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // FP16 support requires compute capability >= 7.0 (Volta, Turing, Ampere, Ada, Hopper)
    int major = prop.major;
    int minor = prop.minor;
    float compute_capability = major + minor * 0.1f;
    
    bool supports = (compute_capability >= 7.0f);
    
    if (supports) {
        printf("GPU: %s (Compute Capability: %d.%d) - FP16 supported\n", 
               prop.name, major, minor);
    } else {
        printf("GPU: %s (Compute Capability: %d.%d) - FP16 NOT supported (requires >= 7.0)\n", 
               prop.name, major, minor);
    }
    
    return supports;
}

// ============================================
// OPTIMIZED FORWARD PASS (Fused kernels + heuristic)
// ============================================
void AutoencoderGPUOptimized::forward(const float* d_input, float* d_recon, int actual_N, cudaStream_t stream) {
    int N = (actual_N > 0) ? actual_N : N_;
    int H16 = H_ / 2;
    int W16 = W_ / 2;
    int H8 = H_ / 4;
    int W8 = W_ / 4;
    
    // FIX: Optimized kernel selection based on analysis
    bool use_fp16 = (use_mixed_precision_ && gpu_supports_fp16_);
    
    // ENCODER
    // Conv1: FP32 fused (C_in=3, rows=27 → fused faster than GEMM)
    conv2d_relu_forward_gpu_fused(
        d_input, d_w1_, d_b1_, d_relu1_,
        N, C_in_, H_, W_, C1_, K_, stream);
    
    maxpool2d_forward_gpu_optimized(d_relu1_, d_pool1_, N, C1_, H_, W_, stream);
    
    // Conv2: FP16/FP32 GEMM (rows=2304 large, GEMM efficient)
    if (use_fp16) {
        conv2d_relu_forward_gemm_fp16(
            d_pool1_, d_w2_, d_b2_, d_relu2_,
            N, C1_, H16, W16, C2_, K_,
            d_im2col_fp16_, d_gemm_out_, cublas_handle_, stream);
    } else {
        conv2d_relu_forward_gemm(
            d_pool1_, d_w2_, d_b2_, d_relu2_,
            N, C1_, H16, W16, C2_, K_,
            d_im2col_, d_gemm_out_, cublas_handle_, stream);
    }
    
    maxpool2d_forward_gpu_optimized(d_relu2_, d_pool2_, N, C2_, H16, W16, stream);
    
    // DECODER
    if (use_fp16) {
        conv2d_relu_forward_gemm_fp16(
            d_pool2_, d_w3_, d_b3_, d_relu3_,
            N, C2_, H8, W8, C3_, K_,
            d_im2col_fp16_, d_gemm_out_, cublas_handle_, stream);
    } else {
        conv2d_relu_forward_gemm(
            d_pool2_, d_w3_, d_b3_, d_relu3_,
            N, C2_, H8, W8, C3_, K_,
            d_im2col_, d_gemm_out_, cublas_handle_, stream);
    }
    
    upsample2d_forward_gpu_optimized(d_relu3_, d_up1_, N, C3_, H8, W8, stream);
    
    if (use_fp16) {
        conv2d_relu_forward_gemm_fp16(
            d_up1_, d_w4_, d_b4_, d_relu4_,
            N, C3_, H16, W16, C4_, K_,
            d_im2col_fp16_, d_gemm_out_, cublas_handle_, stream);
    } else {
        conv2d_relu_forward_gemm(
            d_up1_, d_w4_, d_b4_, d_relu4_,
            N, C3_, H16, W16, C4_, K_,
            d_im2col_, d_gemm_out_, cublas_handle_, stream);
    }
    
    upsample2d_forward_gpu_optimized(d_relu4_, d_up2_, N, C4_, H16, W16, stream);
    
    // Conv5: giữ naive (3 output channels)
    conv2d_forward_gpu_naive(
        d_up2_, d_w5_, d_b5_, d_recon,
        N, C4_, H_, W_, C5_, K_, stream);
}

// ============================================
// FEATURE EXTRACTION (Encoder only) - FULLY OPTIMIZED
// ============================================
void AutoencoderGPUOptimized::extract_features(const float* d_input, float* d_features, int actual_N, cudaStream_t stream) {
    int N = (actual_N > 0) ? actual_N : N_;
    int H16 = H_ / 2;
    int W16 = W_ / 2;
    int H8 = H_ / 4;
    int W8 = W_ / 4;
    
    bool use_fp16 = (use_mixed_precision_ && gpu_supports_fp16_);
    
    // Conv1: FP32 fused
    conv2d_relu_forward_gpu_fused(
        d_input, d_w1_, d_b1_, d_relu1_,
        N, C_in_, H_, W_, C1_, K_, stream);
    
    maxpool2d_forward_gpu_optimized(d_relu1_, d_pool1_, N, C1_, H_, W_, stream);
    
    // Conv2: FP16 GEMM nếu GPU supports
    if (use_fp16) {
        conv2d_relu_forward_gemm_fp16(
            d_pool1_, d_w2_, d_b2_, d_relu2_,
            N, C1_, H16, W16, C2_, K_,
            d_im2col_fp16_, d_gemm_out_, cublas_handle_, stream);
    } else {
        conv2d_relu_forward_gemm(
            d_pool1_, d_w2_, d_b2_, d_relu2_,
            N, C1_, H16, W16, C2_, K_,
            d_im2col_, d_gemm_out_, cublas_handle_, stream);
    }
    
    maxpool2d_forward_gpu_optimized(d_relu2_, d_pool2_, N, C2_, H16, W16, stream);
    
    // Copy latent representation to output
    int latent_size = N * C2_ * H8 * W8;
    CUDA_CHECK(cudaMemcpyAsync(d_features, d_pool2_, latent_size * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
}

// ============================================
// TRAINING STEP (Forward + Backward + Update)
// ============================================
float AutoencoderGPUOptimized::train_step(const float* d_input, float* d_recon,
                                          int actual_N,
                                          cudaStream_t stream,
                                          bool compute_loss_host,
                                          float* h_loss_out) {
    int N = (actual_N > 0) ? actual_N : N_;
    forward(d_input, d_recon, N, stream);
    int total = N * C5_ * H_ * W_;
    mse_loss_backward_gpu(d_recon, d_input, d_drecon_, total, stream);
    backward(d_input, d_recon, d_drecon_, N, stream);
    step(stream);
    float loss_host = 0.0f;
    if (compute_loss_host) {
        loss_host = mse_loss_forward_gpu(d_recon, d_input, total, stream);
        if (h_loss_out) *h_loss_out = loss_host;
    }
    return loss_host;
}

// Async loss version: loss computed on separate stream
void AutoencoderGPUOptimized::train_step_async_loss(const float* d_input, float* d_recon,
                                                    int actual_N,
                                                    cudaStream_t stream_compute,
                                                    float* d_loss_buf,
                                                    float* h_loss_buf,
                                                    cudaEvent_t ev_compute_done,
                                                    cudaEvent_t ev_loss_done,
                                                    cudaStream_t stream_loss) {
    int N = (actual_N > 0) ? actual_N : N_;
    forward(d_input, d_recon, N, stream_compute);
    int total = N * C5_ * H_ * W_;
    mse_loss_backward_gpu(d_recon, d_input, d_drecon_, total, stream_compute);
    backward(d_input, d_recon, d_drecon_, N, stream_compute);
    step(stream_compute);
    
    // Record event after compute completes
    CUDA_CHECK(cudaEventRecord(ev_compute_done, stream_compute));
    
    // Wait for compute to finish, then compute loss on stream_loss
    CUDA_CHECK(cudaStreamWaitEvent(stream_loss, ev_compute_done, 0));
    
    // Zero loss buffer
    CUDA_CHECK(cudaMemsetAsync(d_loss_buf, 0, sizeof(float), stream_loss));
    
    // Compute loss async
    mse_loss_forward_gpu_async(d_recon, d_input, total, d_loss_buf, h_loss_buf, stream_loss);
    
    // Record event when loss copy completes
    CUDA_CHECK(cudaEventRecord(ev_loss_done, stream_loss));
}


// ============================================
// OPTIMIZED KERNELS: Batch Operations
// ============================================

// Kernel để zero tất cả gradients cùng lúc (thay vì 10 memset calls)
__global__ void zero_gradients_batched_kernel(
    float* d_dw1, int n_w1,
    float* d_db1, int n_b1,
    float* d_dw2, int n_w2,
    float* d_db2, int n_b2,
    float* d_dw3, int n_w3,
    float* d_db3, int n_b3,
    float* d_dw4, int n_w4,
    float* d_db4, int n_b4,
    float* d_dw5, int n_w5,
    float* d_db5, int n_b5)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_w1 + n_b1 + n_w2 + n_b2 + n_w3 + n_b3 + n_w4 + n_b4 + n_w5 + n_b5;
    
    if (idx >= total) return;
    
    // Map global index to specific buffer
    int offset = 0;
    if (idx < n_w1) { d_dw1[idx] = 0.0f; return; }
    offset += n_w1;
    if (idx < offset + n_b1) { d_db1[idx - offset] = 0.0f; return; }
    offset += n_b1;
    if (idx < offset + n_w2) { d_dw2[idx - offset] = 0.0f; return; }
    offset += n_w2;
    if (idx < offset + n_b2) { d_db2[idx - offset] = 0.0f; return; }
    offset += n_b2;
    if (idx < offset + n_w3) { d_dw3[idx - offset] = 0.0f; return; }
    offset += n_w3;
    if (idx < offset + n_b3) { d_db3[idx - offset] = 0.0f; return; }
    offset += n_b3;
    if (idx < offset + n_w4) { d_dw4[idx - offset] = 0.0f; return; }
    offset += n_w4;
    if (idx < offset + n_b4) { d_db4[idx - offset] = 0.0f; return; }
    offset += n_b4;
    if (idx < offset + n_w5) { d_dw5[idx - offset] = 0.0f; return; }
    offset += n_w5;
    if (idx < offset + n_b5) { d_db5[idx - offset] = 0.0f; }
}

// Batched SGD update kernel (thay vì 10 kernel launches riêng biệt)
__global__ void sgd_update_batched_kernel(
    float* d_w1, const float* d_dw1, int n_w1,
    float* d_b1, const float* d_db1, int n_b1,
    float* d_w2, const float* d_dw2, int n_w2,
    float* d_b2, const float* d_db2, int n_b2,
    float* d_w3, const float* d_dw3, int n_w3,
    float* d_b3, const float* d_db3, int n_b3,
    float* d_w4, const float* d_dw4, int n_w4,
    float* d_b4, const float* d_db4, int n_b4,
    float* d_w5, const float* d_dw5, int n_w5,
    float* d_b5, const float* d_db5, int n_b5,
    float lr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_w1 + n_b1 + n_w2 + n_b2 + n_w3 + n_b3 + n_w4 + n_b4 + n_w5 + n_b5;
    
    if (idx >= total) return;
    
    // Map global index to specific buffer và update
    int offset = 0;
    if (idx < n_w1) { d_w1[idx] -= lr * d_dw1[idx]; return; }
    offset += n_w1;
    if (idx < offset + n_b1) { d_b1[idx - offset] -= lr * d_db1[idx - offset]; return; }
    offset += n_b1;
    if (idx < offset + n_w2) { d_w2[idx - offset] -= lr * d_dw2[idx - offset]; return; }
    offset += n_w2;
    if (idx < offset + n_b2) { d_b2[idx - offset] -= lr * d_db2[idx - offset]; return; }
    offset += n_b2;
    if (idx < offset + n_w3) { d_w3[idx - offset] -= lr * d_dw3[idx - offset]; return; }
    offset += n_w3;
    if (idx < offset + n_b3) { d_b3[idx - offset] -= lr * d_db3[idx - offset]; return; }
    offset += n_b3;
    if (idx < offset + n_w4) { d_w4[idx - offset] -= lr * d_dw4[idx - offset]; return; }
    offset += n_w4;
    if (idx < offset + n_b4) { d_b4[idx - offset] -= lr * d_db4[idx - offset]; return; }
    offset += n_b4;
    if (idx < offset + n_w5) { d_w5[idx - offset] -= lr * d_dw5[idx - offset]; return; }
    offset += n_w5;
    if (idx < offset + n_b5) { d_b5[idx - offset] -= lr * d_db5[idx - offset]; }
}

// ============================================
// BACKWARD PASS (All gradients) - corrected to accept actual_N
// ============================================
void AutoencoderGPUOptimized::backward(const float* d_input,
                                       const float* d_recon,
                                       const float* d_drecon,
                                       int actual_N,
                                       cudaStream_t stream) {
    int N = (actual_N > 0) ? actual_N : N_;
    int H16 = H_ / 2;
    int W16 = W_ / 2;
    int H8 = H_ / 4;
    int W8 = W_ / 4;
    
    // OPTIMIZATION: Zero tất cả gradients trong một kernel launch thay vì 10 memset calls
    int n_w1 = C1_ * C_in_ * K_ * K_;
    int n_w2 = C2_ * C1_ * K_ * K_;
    int n_w3 = C3_ * C2_ * K_ * K_;
    int n_w4 = C4_ * C3_ * K_ * K_;
    int n_w5 = C5_ * C4_ * K_ * K_;
    int total_grads = n_w1 + C1_ + n_w2 + C2_ + n_w3 + C3_ + n_w4 + C4_ + n_w5 + C5_;
    int block_size = 256;
    int grid_size = (total_grads + block_size - 1) / block_size;
    zero_gradients_batched_kernel<<<grid_size, block_size, 0, stream>>>(
        d_dw1_, n_w1,
        d_db1_, C1_,
        d_dw2_, n_w2,
        d_db2_, C2_,
        d_dw3_, n_w3,
        d_db3_, C3_,
        d_dw4_, n_w4,
        d_db4_, C4_,
        d_dw5_, n_w5,
        d_db5_, C5_);
    CUDA_CHECK(cudaGetLastError());

    // DECODER BACKWARD
    conv2d_backward_gpu_optimized(
        d_drecon, d_up2_, d_w5_,
        d_dup2_, d_dw5_, d_db5_,
        N, C4_, H_, W_, C5_, K_, stream);
    CUDA_CHECK(cudaGetLastError());
    upsample2d_backward_gpu(d_dup2_, d_drelu4_, N, C4_, H16, W16, stream);
    CUDA_CHECK(cudaGetLastError());
    relu_backward_gpu(d_drelu4_, d_relu4_, d_dconv4_, N * C4_ * H16 * W16, stream);
    CUDA_CHECK(cudaGetLastError());
    conv2d_backward_gpu_gemm(
        d_dconv4_, d_up1_, d_w4_,
        d_dup1_, d_dw4_, d_db4_,
        N, C3_, H16, W16, C4_, K_,
        d_im2col_, cublas_handle_, stream);
    CUDA_CHECK(cudaGetLastError());
    upsample2d_backward_gpu(d_dup1_, d_drelu3_, N, C3_, H8, W8, stream);
    CUDA_CHECK(cudaGetLastError());
    relu_backward_gpu(d_drelu3_, d_relu3_, d_dconv3_, N * C3_ * H8 * W8, stream);
    CUDA_CHECK(cudaGetLastError());
    conv2d_backward_gpu_gemm(
        d_dconv3_, d_pool2_, d_w3_,
        d_dpool2_, d_dw3_, d_db3_,
        N, C2_, H8, W8, C3_, K_,
        d_im2col_, cublas_handle_, stream);
    CUDA_CHECK(cudaGetLastError());

    // ENCODER BACKWARD
    maxpool2d_backward_gpu(d_dpool2_, d_relu2_, d_drelu2_, N, C2_, H16, W16, stream);
    CUDA_CHECK(cudaGetLastError());
    relu_backward_gpu(d_drelu2_, d_relu2_, d_dconv2_, N * C2_ * H16 * W16, stream);
    CUDA_CHECK(cudaGetLastError());
    conv2d_backward_gpu_gemm(
        d_dconv2_, d_pool1_, d_w2_,
        d_dpool1_, d_dw2_, d_db2_,
        N, C1_, H16, W16, C2_, K_,
        d_im2col_, cublas_handle_, stream);
    CUDA_CHECK(cudaGetLastError());
    maxpool2d_backward_gpu(d_dpool1_, d_relu1_, d_drelu1_, N, C1_, H_, W_, stream);
    CUDA_CHECK(cudaGetLastError());
    relu_backward_gpu(d_drelu1_, d_relu1_, d_dconv1_, N * C1_ * H_ * W_, stream);
    CUDA_CHECK(cudaGetLastError());
    // Conv1 backward using optimized kernel
    conv2d_backward_gpu_optimized(
        d_dconv1_, d_input, d_w1_,
        d_dinput_temp_, d_dw1_, d_db1_,
        N, C_in_, H_, W_, C1_, K_, stream);
    CUDA_CHECK(cudaGetLastError());
}

// Legacy SGD kernel (giữ lại cho compatibility)
__global__ void sgd_update_kernel(float* param, const float* grad, float lr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    param[idx] -= lr * grad[idx];
}

void AutoencoderGPUOptimized::step(cudaStream_t stream) {
    // OPTIMIZATION: Batched SGD update trong một kernel launch thay vì 10 launches riêng biệt
    int n_w1 = C1_ * C_in_ * K_ * K_;
    int n_w2 = C2_ * C1_ * K_ * K_;
    int n_w3 = C3_ * C2_ * K_ * K_;
    int n_w4 = C4_ * C3_ * K_ * K_;
    int n_w5 = C5_ * C4_ * K_ * K_;
    int total_params = n_w1 + C1_ + n_w2 + C2_ + n_w3 + C3_ + n_w4 + C4_ + n_w5 + C5_;
    int block_size = 256;
    int grid_size = (total_params + block_size - 1) / block_size;
    
    sgd_update_batched_kernel<<<grid_size, block_size, 0, stream>>>(
        d_w1_, d_dw1_, n_w1,
        d_b1_, d_db1_, C1_,
        d_w2_, d_dw2_, n_w2,
        d_b2_, d_db2_, C2_,
        d_w3_, d_dw3_, n_w3,
        d_b3_, d_db3_, C3_,
        d_w4_, d_dw4_, n_w4,
        d_b4_, d_db4_, C4_,
        d_w5_, d_dw5_, n_w5,
        d_b5_, d_db5_, C5_,
        lr_);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// SAVE/LOAD WEIGHTS
// ============================================
void AutoencoderGPUOptimized::save_weights(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        printf("ERROR: Cannot save weights to %s\n", filepath.c_str());
        return;
    }
    
    // FIX: Add header to match Phase 2 format for compatibility
    int num_layers = 5;
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
    
    int n_w1 = C1_ * C_in_ * K_ * K_;
    int n_w2 = C2_ * C1_ * K_ * K_;
    int n_w3 = C3_ * C2_ * K_ * K_;
    int n_w4 = C4_ * C3_ * K_ * K_;
    int n_w5 = C5_ * C4_ * K_ * K_;
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
    
    // Write with size headers (match Phase 2 format)
    file.write(reinterpret_cast<const char*>(&n_w1), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w1.data()), n_w1 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b1.data()), C1_ * sizeof(float));
    
    file.write(reinterpret_cast<const char*>(&n_w2), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w2.data()), n_w2 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b2.data()), C2_ * sizeof(float));
    
    file.write(reinterpret_cast<const char*>(&n_w3), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w3.data()), n_w3 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b3.data()), C3_ * sizeof(float));
    
    file.write(reinterpret_cast<const char*>(&n_w4), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w4.data()), n_w4 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b4.data()), C4_ * sizeof(float));
    
    file.write(reinterpret_cast<const char*>(&n_w5), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_w5.data()), n_w5 * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_b5.data()), C5_ * sizeof(float));
    
    file.close();
}

void AutoencoderGPUOptimized::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        printf("ERROR: Cannot load weights from %s\n", filepath.c_str());
        return;
    }
    
    // FIX: Read header to match Phase 2 format (backward compatible)
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    if (num_layers != 5) {
        printf("ERROR: File has %d layers, expected 5\n", num_layers);
        file.close();
        return;
    }
    
    int n_w1 = C1_ * C_in_ * K_ * K_;
    int n_w2 = C2_ * C1_ * K_ * K_;
    int n_w3 = C3_ * C2_ * K_ * K_;
    int n_w4 = C4_ * C3_ * K_ * K_;
    int n_w5 = C5_ * C4_ * K_ * K_;
    std::vector<float> h_w1(n_w1), h_w2(n_w2), h_w3(n_w3), h_w4(n_w4), h_w5(n_w5);
    std::vector<float> h_b1(C1_), h_b2(C2_), h_b3(C3_), h_b4(C4_), h_b5(C5_);
    
    // Read with size verification (match Phase 2 format)
    int n_read;
    
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    if (n_read != n_w1) {
        printf("ERROR: Layer 1 weight size mismatch: got %d, expected %d\n", n_read, n_w1);
        file.close();
        return;
    }
    file.read(reinterpret_cast<char*>(h_w1.data()), n_w1 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b1.data()), C1_ * sizeof(float));
    
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    if (n_read != n_w2) {
        printf("ERROR: Layer 2 weight size mismatch: got %d, expected %d\n", n_read, n_w2);
        file.close();
        return;
    }
    file.read(reinterpret_cast<char*>(h_w2.data()), n_w2 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b2.data()), C2_ * sizeof(float));
    
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    if (n_read != n_w3) {
        printf("ERROR: Layer 3 weight size mismatch: got %d, expected %d\n", n_read, n_w3);
        file.close();
        return;
    }
    file.read(reinterpret_cast<char*>(h_w3.data()), n_w3 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b3.data()), C3_ * sizeof(float));
    
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    if (n_read != n_w4) {
        printf("ERROR: Layer 4 weight size mismatch: got %d, expected %d\n", n_read, n_w4);
        file.close();
        return;
    }
    file.read(reinterpret_cast<char*>(h_w4.data()), n_w4 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b4.data()), C4_ * sizeof(float));
    
    file.read(reinterpret_cast<char*>(&n_read), sizeof(int));
    if (n_read != n_w5) {
        printf("ERROR: Layer 5 weight size mismatch: got %d, expected %d\n", n_read, n_w5);
        file.close();
        return;
    }
    file.read(reinterpret_cast<char*>(h_w5.data()), n_w5 * sizeof(float));
    file.read(reinterpret_cast<char*>(h_b5.data()), C5_ * sizeof(float));
    
    file.close();
    
    // Copy to device
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
}