#ifndef LAYERS_GPU_H
#define LAYERS_GPU_H

// Forward declarations for GPU layers
void conv2d_forward_gpu_naive(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K);

void relu_forward_gpu(
    const float* d_input,
    float* d_output,
    int total_elements);

#endif
