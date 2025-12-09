#ifndef LAYERS_GPU_H
#define LAYERS_GPU_H

// Conv2D forward (naive)
void conv2d_forward_gpu_naive(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream = 0);

// Conv2D backward (naive)
void conv2d_backward_gpu_naive(
    const float* d_out,      // dL/dY
    const float* d_input,    // X
    const float* d_weight,   // W
    float* d_dinput,         // dL/dX
    float* d_dweight,        // dL/dW
    float* d_dbias,          // dL/db
    int N, int C_in, int H, int W,
    int C_out, int K);

// ReLU forward (element-wise)
void relu_forward_gpu(
    const float* d_input,
    float* d_output,
    int total_elements,
    cudaStream_t stream = 0);

// ReLU backward (element-wise)
void relu_backward_gpu(
    const float* d_out,
    const float* d_input,
    float* d_dinput,
    int total_elements,
    cudaStream_t stream = 0);

// MaxPool 2x2, stride=2 forward
void maxpool2d_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    cudaStream_t stream = 0);

// MaxPool 2x2, stride=2 backward
// d_out: (N, C, H/2, W/2)
// input: (N, C, H, W)  -- giá trị trước pool, dùng để tìm vị trí max
// d_dinput: (N, C, H, W) -- gradient wrt input
void maxpool2d_backward_gpu(
    const float* d_out,
    const float* d_input,
    float* d_dinput,
    int N, int C, int H, int W,
    cudaStream_t stream = 0);

// Upsample 2x (nearest neighbor) forward
void upsample2d_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    cudaStream_t stream = 0);

// Upsample 2x backward
void upsample2d_backward_gpu(
    const float* d_out,
    float* d_dinput,
    int N, int C, int H, int W,
    cudaStream_t stream = 0);

// MSE Loss forward: trả về loss (float)
float mse_loss_forward_gpu(
    const float* d_pred,
    const float* d_target,
    int total_elements,
    cudaStream_t stream = 0);

// MSE Loss backward: gradient theo pred
void mse_loss_backward_gpu(
    const float* d_pred,
    const float* d_target,
    float* d_dpred,
    int total_elements,
    cudaStream_t stream = 0);

#endif