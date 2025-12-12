#ifndef LAYERS_GPU_OPTIMIZED_H
#define LAYERS_GPU_OPTIMIZED_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For __half type (FP16 support)
#include <cublas_v2.h>

// ============================================
// Phase 3: Optimized GPU Kernels
// ============================================

// OPTIMIZATION 1: Kernel Fusion (Conv + ReLU + Bias)
void conv2d_relu_forward_gpu_fused(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream = 0,
    bool use_constant_memory = false);  // Training: false (use global), Inference: true (use constant)

// GEMM path: im2col + cuBLAS + bias+ReLU (FP32)
void conv2d_relu_forward_gemm(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    float* d_im2col,        // workspace: C_in*K*K x (N*H*W)
    float* d_gemm_out,      // workspace: C_out x (N*H*W) (column-major)
    cublasHandle_t handle,
    cudaStream_t stream = 0);

// SMART KERNEL SELECTION - Auto-select best kernel based on size
// Automatically chooses between fused, GEMM FP32, or GEMM FP16
void conv2d_relu_forward_smart(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    float* d_im2col,        // workspace FP32
    float* d_gemm_out,      // workspace FP32
    __half* d_im2col_fp16,  // workspace FP16 (can be nullptr if not using FP16)
    cublasHandle_t handle,
    bool use_fp16,          // whether to use FP16 GEMM if available
    cudaStream_t stream = 0);

// GEMM path FP16 activations (Tensor Cores), FP32 weights/accum
void conv2d_relu_forward_gemm_fp16(
    const float* d_input,           // FP32 input
    const float* d_weight,          // FP32 weights
    const float* d_bias,            // FP32 bias
    float* d_output,                // FP32 output
    int N, int C_in, int H, int W,
    int C_out, int K,
    __half* d_im2col_fp16,          // workspace FP16: C_in*K*K x (N*H*W)
    float* d_gemm_out_fp32,         // workspace FP32: C_out x (N*H*W)
    cublasHandle_t handle,
    cudaStream_t stream = 0);

// Backward via im2col + cuBLAS GEMM (Conv only, no activation)
void conv2d_backward_gpu_gemm(
    const float* d_out,       // (N, C_out, H, W)
    const float* d_input,     // (N, C_in, H, W)
    const float* d_weight,    // (C_out, C_in, K, K)
    float* d_dinput,          // (N, C_in, H, W) - zeroed inside
    float* d_dweight,         // (C_out, C_in, K, K)
    float* d_dbias,           // (C_out)
    int N, int C_in, int H, int W,
    int C_out, int K,
    float* d_im2col,          // workspace reused
    cublasHandle_t handle,
    cudaStream_t stream = 0);

// OPTIMIZATION 2: Shared Memory Tiling
void conv2d_forward_gpu_tiled(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream = 0);

// OPTIMIZATION 3: Optimized Backward
void conv2d_backward_gpu_optimized(
    const float* d_out,
    const float* d_input,
    const float* d_weight,
    float* d_dinput,
    float* d_dweight,
    float* d_dbias,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream = 0);

// OPTIMIZATION 4: Pinned Memory Helpers
float* allocate_pinned_memory(size_t size);
void free_pinned_memory(float* ptr);

// OPTIMIZATION 5: Vectorized ReLU (with stream support)
void relu_forward_gpu_vectorized(const float* d_input, float* d_output, int total, cudaStream_t stream = 0);

// Smart wrapper: Auto-selects vectorized or scalar ReLU based on size
void relu_forward_gpu_smart(const float* d_input, float* d_output, int total, cudaStream_t stream = 0);

// OPTIMIZATION 6: Vectorized Memory Access (float4) for Convolution
void conv2d_relu_forward_gpu_fused_vectorized(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream = 0);

// OPTIMIZATION 7: Optimized Pooling với shared memory và coalesced access
void maxpool2d_forward_gpu_optimized(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    cudaStream_t stream = 0);

// OPTIMIZATION 8: Optimized Upsampling với vectorized access
void upsample2d_forward_gpu_optimized(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    cudaStream_t stream = 0);

// Constant memory helper: Update biases in constant memory
// Layout: [b1(256), b2(128), b3(128), b4(256), b5(3)] = 771 floats
void update_constant_memory_biases(
    const float* d_b1, int c1,
    const float* d_b2, int c2,
    const float* d_b3, int c3,
    const float* d_b4, int c4,
    const float* d_b5, int c5,
    cudaStream_t stream = 0);

// OPTIMIZATION 13: Mixed Precision (FP16/FP32)
// FP16 activations for forward pass, FP32 for weights
// NOTE: Requires pre-allocated FP16 buffers (reuse across calls for performance)
void conv2d_relu_forward_gpu_fused_fp16(
    const float* d_input_fp32,
    const float* d_weight,
    const float* d_bias,
    float* d_output_fp32,
    __half* d_input_fp16_buffer,   // Pre-allocated FP16 input buffer
    __half* d_output_fp16_buffer,  // Pre-allocated FP16 output buffer
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream = 0);

#endif // LAYERS_GPU_OPTIMIZED_H

