#include <cuda_runtime.h>
#include <cstdio>
#include "layers_gpu.h"

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = cmd; \
        if (e != cudaSuccess) { \
            printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        } \
    } while (0)

// =======================
// CONV2D FORWARD - NAIVE
// =======================

__global__ void conv2d_forward_naive_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    int n = blockIdx.z;
    int co = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;

    if (idx >= HW) return;

    int h = idx / W;
    int w = idx % W;

    int pad = K / 2;

    float sum = 0.0f;

    for (int ci = 0; ci < C_in; ci++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {

                int ih = h + kh - pad;
                int iw = w + kw - pad;

                if (ih < 0 || iw < 0 || ih >= H || iw >= W)
                    continue;

                int input_idx =
                    ((n * C_in + ci) * H + ih) * W + iw;

                int weight_idx =
                    ((co * C_in + ci) * K + kh) * K + kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (bias)
        sum += bias[co];

    int out_idx =
        ((n * C_out + co) * H + h) * W + w;

    output[out_idx] = sum;
}

void conv2d_forward_gpu_naive(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    int HW = H * W;
    dim3 block(256);
    dim3 grid((HW + block.x - 1) / block.x, C_out, N);

    conv2d_forward_naive_kernel<<<grid, block>>>(
        d_input, d_weight, d_bias, d_output,
        N, C_in, H, W, C_out, K);

    CUDA_CHECK(cudaGetLastError());
}
// ===========
// RELU FORWARD
// ===========

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = input[idx];
    output[idx] = x > 0 ? x : 0;
}

void relu_forward_gpu(const float* d_input, float* d_output, int total) {
    int block = 256;
    int grid = (total + block - 1) / block;

    relu_kernel<<<grid, block>>>(d_input, d_output, total);

    CUDA_CHECK(cudaGetLastError());
}

