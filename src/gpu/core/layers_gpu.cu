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
    int n  = blockIdx.z;
    int co = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;

    if (idx >= HW) return;

    int h = idx / W;
    int w = idx % W;

    int pad = K / 2;
    float sum = 0.0f;

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;

                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
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
    int C_out, int K,
    cudaStream_t stream)
{
    int HW = H * W;
    dim3 block(256);
    dim3 grid((HW + block.x - 1) / block.x, C_out, N);

    conv2d_forward_naive_kernel<<<grid, block, 0, stream>>>(
        d_input, d_weight, d_bias, d_output,
        N, C_in, H, W, C_out, K);

    CUDA_CHECK(cudaGetLastError());
}

// =======================
// CONV2D BACKWARD - NAIVE
// =======================

// d_input kernel: mỗi thread xử lý 1 phần tử X[n, ci, h, w]
__global__ void conv2d_backward_input_kernel(
    const float* d_out,      // (N, C_out, H, W)
    const float* weight,     // (C_out, C_in, K, K)
    float* d_input,          // (N, C_in, H, W)
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H * W;
    if (idx >= total) return;

    int HW = H * W;
    int n  = idx / (C_in * HW);
    int rem = idx % (C_in * HW);
    int ci = rem / HW;
    int rem2 = rem % HW;
    int h  = rem2 / W;
    int w  = rem2 % W;

    int pad = K / 2;
    float sum = 0.0f;

    // sum over tất cả output channel & kernel positions
    // Convolution backward input: flip kernel 180° (rotation)
    for (int co = 0; co < C_out; ++co) {
        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                int h_out = h + kh - pad;  // Flip kernel: +kh instead of -kh
                int w_out = w + kw - pad;  // Flip kernel: +kw instead of -kw

                if (h_out < 0 || h_out >= H || w_out < 0 || w_out >= W)
                    continue;

                int out_idx =
                    ((n * C_out + co) * H + h_out) * W + w_out;

                int w_idx =
                    ((co * C_in + ci) * K + kh) * K + kw;

                sum += d_out[out_idx] * weight[w_idx];
            }
        }
    }

    d_input[idx] = sum;
}

// d_weight kernel: mỗi thread xử lý 1 phần tử W[co, ci, kh, kw]
__global__ void conv2d_backward_weight_kernel(
    const float* d_out,    // (N, C_out, H, W)
    const float* input,    // (N, C_in, H, W)
    float* d_weight,       // (C_out, C_in, K, K)
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * K * K;
    if (idx >= total) return;

    int K2 = K * K;
    int co = idx / (C_in * K2);
    int rem = idx % (C_in * K2);
    int ci = rem / K2;
    int rem2 = rem % K2;
    int kh = rem2 / K;
    int kw = rem2 % K;

    int pad = K / 2;
    float sum = 0.0f;

    // sum_n,h,w d_out * corresponding input
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;

                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                    continue;

                int out_idx =
                    ((n * C_out + co) * H + h) * W + w;

                int in_idx =
                    ((n * C_in + ci) * H + ih) * W + iw;

                sum += d_out[out_idx] * input[in_idx];
            }
        }
    }

    d_weight[idx] = sum;
}

// d_bias kernel: mỗi thread xử lý 1 bias[co]
__global__ void conv2d_backward_bias_kernel(
    const float* d_out,  // (N, C_out, H, W)
    float* d_bias,       // (C_out)
    int N, int C_out, int H, int W)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int out_idx =
                    ((n * C_out + co) * H + h) * W + w;
                sum += d_out[out_idx];
            }
        }
    }
    d_bias[co] = sum;
}

void conv2d_backward_gpu_naive(
    const float* d_out,
    const float* d_input,
    const float* d_weight,
    float* d_dinput,
    float* d_dweight,
    float* d_dbias,
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    // Zero gradients trước
    CUDA_CHECK(cudaMemset(d_dinput,  0, N * C_in * H * W      * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dweight, 0, C_out * C_in * K * K  * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dbias,   0, C_out                 * sizeof(float)));

    // d_input
    {
        int total_in = N * C_in * H * W;
        int block = 256;
        int grid  = (total_in + block - 1) / block;
        conv2d_backward_input_kernel<<<grid, block>>>(
            d_out, d_weight, d_dinput,
            N, C_in, H, W, C_out, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // d_weight
    {
        int total_w = C_out * C_in * K * K;
        int block = 256;
        int grid  = (total_w + block - 1) / block;
        conv2d_backward_weight_kernel<<<grid, block>>>(
            d_out, d_input, d_dweight,
            N, C_in, H, W, C_out, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // d_bias
    {
        int block = 256;
        int grid  = (C_out + block - 1) / block;
        conv2d_backward_bias_kernel<<<grid, block>>>(
            d_out, d_dbias,
            N, C_out, H, W);
        CUDA_CHECK(cudaGetLastError());
    }
}


// ===========
// RELU FORWARD
// ===========

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = input[idx];
    output[idx] = x > 0.0f ? x : 0.0f;
}

void relu_forward_gpu(const float* d_input, float* d_output, int total, cudaStream_t stream) {
    int block = 256;
    int grid  = (total + block - 1) / block;

    relu_kernel<<<grid, block, 0, stream>>>(d_input, d_output, total);

    CUDA_CHECK(cudaGetLastError());
}

// ===========
// RELU BACKWARD
// ===========

__global__ void relu_backward_kernel(
    const float* d_out,
    const float* input,
    float* d_input,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = input[idx];
    float grad = d_out[idx];
    d_input[idx] = (x > 0.0f) ? grad : 0.0f;
}

void relu_backward_gpu(
    const float* d_out,
    const float* d_input_forward,
    float* d_dinput,
    int total_elements,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (total_elements + block - 1) / block;

    relu_backward_kernel<<<grid, block, 0, stream>>>(
        d_out, d_input_forward, d_dinput, total_elements);

    CUDA_CHECK(cudaGetLastError());
}

// ======================
// MAXPOOL 2x2, STRIDE 2 FORWARD
// ======================

__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W)
{
    int n  = blockIdx.z;
    int c  = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H / 2;
    int W_out = W / 2;
    int HW_out = H_out * W_out;

    if (idx >= HW_out) return;

    int h_out = idx / W_out;
    int w_out = idx % W_out;

    int h_in = h_out * 2;
    int w_in = w_out * 2;

    auto in_idx = [C, H, W](int n, int c, int h, int w) {
        return ((n * C + c) * H + h) * W + w;
    };

    float m = -1e30f;

    #pragma unroll
    for (int dh = 0; dh < 2; ++dh) {
        #pragma unroll
        for (int dw = 0; dw < 2; ++dw) {
            int ih = h_in + dh;
            int iw = w_in + dw;

            if (ih >= H || iw >= W) continue;

            float v = input[in_idx(n, c, ih, iw)];
            if (v > m) m = v;
        }
    }

    int out_idx =
        ((n * C + c) * H_out + h_out) * W_out + w_out;

    output[out_idx] = m;
}

void maxpool2d_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int H_out = H / 2;
    int W_out = W / 2;
    int HW_out = H_out * W_out;

    dim3 block(256);
    dim3 grid((HW_out + block.x - 1) / block.x, C, N);

    maxpool2d_forward_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, N, C, H, W);

    CUDA_CHECK(cudaGetLastError());
}

// ======================
// MAXPOOL 2x2, STRIDE 2 BACKWARD
// ======================

__global__ void maxpool2d_backward_kernel(
    const float* d_out,
    const float* input,
    float* d_input,
    int N, int C, int H, int W)
{
    int n  = blockIdx.z;
    int c  = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H / 2;
    int W_out = W / 2;
    int HW_out = H_out * W_out;

    if (idx >= HW_out) return;

    int h_out = idx / W_out;
    int w_out = idx % W_out;

    int h_in = h_out * 2;
    int w_in = w_out * 2;

    auto in_idx = [C, H, W](int n, int c, int h, int w) {
        return ((n * C + c) * H + h) * W + w;
    };

    // Tìm argmax trong block 2x2 của input
    float max_val = -1e30f;
    int max_h = h_in;
    int max_w = w_in;

    #pragma unroll
    for (int dh = 0; dh < 2; ++dh) {
        #pragma unroll
        for (int dw = 0; dw < 2; ++dw) {
            int ih = h_in + dh;
            int iw = w_in + dw;

            if (ih >= H || iw >= W) continue;

            float v = input[in_idx(n, c, ih, iw)];
            if (v > max_val) {
                max_val = v;
                max_h = ih;
                max_w = iw;
            }
        }
    }

    float grad = d_out[((n * C + c) * H_out + h_out) * W_out + w_out];

    // Vì các block 2x2 không overlap (stride=2) nên mỗi (h,w) chỉ thuộc về 1 block
    int in_index = in_idx(n, c, max_h, max_w);
    d_input[in_index] += grad;
}

void maxpool2d_backward_gpu(
    const float* d_out,
    const float* d_input_forward,
    float* d_dinput,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    // d_dinput phải zero trước khi cộng gradient
    CUDA_CHECK(cudaMemset(d_dinput, 0, N * C * H * W * sizeof(float)));

    int H_out = H / 2;
    int W_out = W / 2;
    int HW_out = H_out * W_out;

    dim3 block(256);
    dim3 grid((HW_out + block.x - 1) / block.x, C, N);

    maxpool2d_backward_kernel<<<grid, block, 0, stream>>>(
        d_out, d_input_forward, d_dinput, N, C, H, W);

    CUDA_CHECK(cudaGetLastError());
}

// ======================
// UPSAMPLE 2x (NEAREST) FORWARD
// ======================

__global__ void upsample2d_forward_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W)
{
    int n  = blockIdx.z;
    int c  = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H * 2;
    int W_out = W * 2;
    int HW_out = H_out * W_out;

    if (idx >= HW_out) return;

    int h_out = idx / W_out;
    int w_out = idx % W_out;

    int h_in = h_out / 2;
    int w_in = w_out / 2;

    int in_idx =
        ((n * C + c) * H + h_in) * W + w_in;

    int out_idx =
        ((n * C + c) * H_out + h_out) * W_out + w_out;

    output[out_idx] = input[in_idx];
}

void upsample2d_forward_gpu(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int H_out = H * 2;
    int W_out = W * 2;
    int HW_out = H_out * W_out;

    dim3 block(256);
    dim3 grid((HW_out + block.x - 1) / block.x, C, N);

    upsample2d_forward_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, N, C, H, W);

    CUDA_CHECK(cudaGetLastError());
}

// ======================
// UPSAMPLE 2x BACKWARD
// ======================

__global__ void upsample2d_backward_kernel(
    const float* d_out,
    float* d_in,
    int N, int C, int H, int W)
{
    // mỗi thread xử lý 1 phần tử input (n,c,h,w)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_in = N * C * H * W;
    if (idx >= total_in) return;

    int HW = H * W;
    int n  = idx / (C * HW);
    int rem = idx % (C * HW);
    int c  = rem / HW;
    int rem2 = rem % HW;
    int h  = rem2 / W;
    int w  = rem2 % W;

    int H_out = H * 2;
    int W_out = W * 2;

    int h_out0 = h * 2;
    int w_out0 = w * 2;

    auto out_idx = [C, H_out, W_out](int n, int c, int h, int w) {
        return ((n * C + c) * H_out + h) * W_out + w;
    };

    float grad = 0.0f;
    grad += d_out[out_idx(n, c, h_out0,     w_out0    )];
    grad += d_out[out_idx(n, c, h_out0 + 1, w_out0    )];
    grad += d_out[out_idx(n, c, h_out0,     w_out0 + 1)];
    grad += d_out[out_idx(n, c, h_out0 + 1, w_out0 + 1)];

    d_in[idx] = grad;
}

void upsample2d_backward_gpu(
    const float* d_out,
    float* d_dinput,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total_in = N * C * H * W;
    int block = 256;
    int grid  = (total_in + block - 1) / block;

    upsample2d_backward_kernel<<<grid, block, 0, stream>>>(
        d_out, d_dinput, N, C, H, W);

    CUDA_CHECK(cudaGetLastError());
}

// ======================
// MSE LOSS FORWARD
// ======================

// Warp-level reduction helper (faster than shared memory)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized warp-level reduction kernel
__global__ void mse_loss_forward_warp_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* loss_accum,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int step = blockDim.x * gridDim.x;

    // Each thread computes local sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += step) {
        float diff = pred[i] - target[i];
        local_sum += diff * diff;
    }

    // Warp-level reduction (no shared memory needed for this step)
    local_sum = warp_reduce_sum(local_sum);

    // First thread in each warp writes to shared memory
    // Max 8 warps per block (256 threads / 32 = 8 warps)
    // Allocate 32 floats để safe cho block size lớn hơn
    __shared__ float warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);

        if (lane_id == 0) {
            atomicAdd(loss_accum, val);
        }
    }
}

// Original shared memory kernel (kept for fallback/comparison)
__global__ void mse_loss_forward_kernel(
    const float* pred,
    const float* target,
    float* loss_accum,
    int N)
{
    extern __shared__ float sdata[];

    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = idx; i < N; i += step) {
        float diff = pred[i] - target[i];
        local_sum += diff * diff;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss_accum, sdata[0]);
    }
}

float mse_loss_forward_gpu(
    const float* d_pred,
    const float* d_target,
    int total_elements,
    cudaStream_t stream)
{
    float h_loss = 0.0f;
    float* d_loss = nullptr;

    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    int block = 256;
    int grid  = (total_elements + block - 1) / block;
    if (grid > 1024) grid = 1024;

    // Use warp-level reduction kernel (faster, less shared memory)
    // Max 8 warps per block (256 threads / 32 = 8 warps)
    // Allocate 32 floats để safe cho block size lớn hơn
    size_t shared_size = 32 * sizeof(float);
    mse_loss_forward_warp_kernel<<<grid, block, shared_size, stream>>>(
        d_pred, d_target, d_loss, total_elements);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));

    h_loss /= static_cast<float>(total_elements);
    return h_loss;
}

// Async version: uses pre-allocated buffers, returns immediately
// Caller must sync stream_loss and divide by total_elements
void mse_loss_forward_gpu_async(
    const float* d_pred,
    const float* d_target,
    int total_elements,
    float* d_loss_buf,  // Pre-allocated device buffer (must be zeroed)
    float* h_loss_buf,  // Host buffer to receive result
    cudaStream_t stream_loss)
{
    int block = 256;
    int grid  = (total_elements + block - 1) / block;
    if (grid > 1024) grid = 1024;

    // Use warp-level reduction kernel (faster, less shared memory)
    // Max 8 warps per block (256 threads / 32 = 8 warps)
    // Allocate 32 floats để safe cho block size lớn hơn
    size_t shared_size = 32 * sizeof(float);
    mse_loss_forward_warp_kernel<<<grid, block, shared_size, stream_loss>>>(
        d_pred, d_target, d_loss_buf, total_elements);

    CUDA_CHECK(cudaGetLastError());

    // Async copy to host
    CUDA_CHECK(cudaMemcpyAsync(h_loss_buf, d_loss_buf, sizeof(float),
                               cudaMemcpyDeviceToHost, stream_loss));
}

// ======================
// MSE LOSS BACKWARD
// ======================

__global__ void mse_loss_backward_kernel(
    const float* pred,
    const float* target,
    float* d_pred,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float diff = pred[idx] - target[idx];
    d_pred[idx] = 2.0f * diff / static_cast<float>(N);
}

void mse_loss_backward_gpu(
    const float* d_pred,
    const float* d_target,
    float* d_dpred,
    int total_elements,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (total_elements + block - 1) / block;

    mse_loss_backward_kernel<<<grid, block, 0, stream>>>(
        d_pred, d_target, d_dpred, total_elements);

    CUDA_CHECK(cudaGetLastError());
}