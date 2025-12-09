#ifndef LAYERS_GPU_OPTIMIZED_H
#define LAYERS_GPU_OPTIMIZED_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For __half type (FP16 support)

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
    cudaStream_t stream = 0);

// OPTIMIZATION 1B: Fused + Tiled + Channel Blocking (Phase 3B - BEST!)
void conv2d_relu_forward_gpu_fused_tiled(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
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

// Optional: Conv forward using constant memory for very small outputs (e.g., Conv5 C_out<=16, C_in<=3)
bool conv2d_forward_gpu_const_small(
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

// Constant Memory Helper: Copy Conv1 weights to constant memory (call once during initialization)
void copy_conv1_weights_to_constant(const float* d_weight, const float* d_bias, int C_out, cudaStream_t stream = 0);

// Constant Memory Helper: Update bias in constant memory (call after SGD step)
void update_conv1_bias_in_constant(const float* d_bias, int C_out, cudaStream_t stream = 0);

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

