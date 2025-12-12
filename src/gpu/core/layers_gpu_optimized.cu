// layers_gpu_optimized.cu - Optimized GPU layers implementation
#include "layers_gpu_optimized.h"
#include "layers_gpu.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For FP16 support
#include <cublas_v2.h>
#include <cstdio>
#include <cstddef>
#include <vector>

// Constant memory for biases
__constant__ float c_bias_all[256 + 128 + 128 + 256 + 3];
#define C_BIAS1_OFFSET 0
#define C_BIAS2_OFFSET 256
#define C_BIAS3_OFFSET 384
#define C_BIAS4_OFFSET 512
#define C_BIAS5_OFFSET 768

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = cmd; \
        if (e != cudaSuccess) { \
            printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e)); \
        } \
    } while (0)

// GEMM path: im2col + cuBLAS + bias + ReLU
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    int N, int C, int H, int W, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    int total_cols = N * HW;
    if (idx >= total_cols) return;

    int n = idx / HW;
    int m = idx % HW;
    int h = m / W;
    int w = m % W;
    int pad = K / 2;

    int K2 = K * K;
    int col_base = idx;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;
                float val = 0.0f;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int im_idx = ((n * C + c) * H + ih) * W + iw;
                    val = data_im[im_idx];
                }
                int k_idx = (c * K2) + kh * K + kw;
                data_col[k_idx * total_cols + col_base] = val;
            }
        }
    }
}

__global__ void gemm_output_bias_relu_kernel(
    const float* __restrict__ gemm_out_col, // shape: C_out x (N*H*W), column-major lda=C_out
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C_out, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C_out;
    if (idx >= total) return;

    int c = idx % C_out;
    int spatial = idx / C_out; // 0 .. N*H*W -1
    int n = spatial / (H * W);
    int hw = spatial % (H * W);
    int h = hw / W;
    int w = hw % W;

    float val = gemm_out_col[c + spatial * C_out] + bias[c];
    val = fmaxf(val, 0.0f);
    int out_idx = ((n * C_out + c) * H + h) * W + w;
    out[out_idx] = val;
}

void conv2d_relu_forward_gemm(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    float* d_im2col,
    float* d_gemm_out,
    cublasHandle_t handle,
    cudaStream_t stream)
{
    int HW = H * W;
    int cols = N * HW;           // number of columns (one per spatial location per batch)
    int rows = C_in * K * K;     // rows of im2col

    // 1) im2col
    int block = 256;
    int grid = (cols + block - 1) / block;
    im2col_kernel<<<grid, block, 0, stream>>>(d_input, d_im2col, N, C_in, H, W, K);

    // 2) GEMM: (C_out x rows) * (rows x cols) => (C_out x cols)
    // Using column-major view:
    // A = d_weight (C_out x rows) row-major -> treat as column-major with ld = rows, opA = N or T?
    // We stored weights as [C_out][rows] contiguous, so row-major. To interpret as column-major,
    // set opA = CUBLAS_OP_T with lda = rows to get (rows x C_out)^(T) = (C_out x rows).
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(handle, stream);
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        C_out,            // m
        cols,             // n
        rows,             // k
        &alpha,
        d_weight,         // A
        rows,             // lda (since opA=T)
        d_im2col,         // B
        rows,             // ldb
        &beta,
        d_gemm_out,       // C (column-major, lda = C_out)
        C_out);

    // 3) Bias + ReLU and reorder to NCHW
    int total = N * C_out * H * W;
    block = 256;
    grid = (total + block - 1) / block;
    gemm_output_bias_relu_kernel<<<grid, block, 0, stream>>>(
        d_gemm_out, d_bias, d_output, N, C_out, H, W);
    CUDA_CHECK(cudaGetLastError());
}

// FP16 im2col kernel: writes half
__global__ void im2col_fp16_kernel(
    const float* __restrict__ data_im,
    __half* __restrict__ data_col,
    int N, int C, int H, int W, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    int total_cols = N * HW;
    if (idx >= total_cols) return;

    int n = idx / HW;
    int m = idx % HW;
    int h = m / W;
    int w = m % W;
    int pad = K / 2;

    int K2 = K * K;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;
                float val = 0.0f;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int im_idx = ((n * C + c) * H + ih) * W + iw;
                    val = data_im[im_idx];
                }
                int k_idx = (c * K2) + kh * K + kw; // [0, C*K*K)
                data_col[k_idx * total_cols + idx] = __float2half(val);
            }
        }
    }
}

void conv2d_relu_forward_gemm_fp16(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    __half* d_im2col_fp16,
    float* d_gemm_out_fp32,
    cublasHandle_t handle,
    cudaStream_t stream)
{
    int HW = H * W;
    int cols = N * HW;
    int rows = C_in * K * K;

    int block = 256;
    int grid = (cols + block - 1) / block;
    im2col_fp16_kernel<<<grid, block, 0, stream>>>(d_input, d_im2col_fp16, N, C_in, H, W, K);

    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(handle, stream);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        C_out,           // m
        cols,            // n
        rows,            // k
        &alpha,
        d_weight,        CUDA_R_32F, rows,   // A
        d_im2col_fp16,   CUDA_R_16F, rows,   // B
        &beta,
        d_gemm_out_fp32, CUDA_R_32F, C_out,  // C
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    int total = N * C_out * H * W;
    grid = (total + block - 1) / block;
    gemm_output_bias_relu_kernel<<<grid, block, 0, stream>>>(
        d_gemm_out_fp32, d_bias, d_output, N, C_out, H, W);
    CUDA_CHECK(cudaGetLastError());
}

// Backward: im2col + GEMM + col2im
// col2im without atomics: each thread produces one (n,c,h,w)
__global__ void col2im_kernel_noatomic(
    const float* __restrict__ data_col, // rows x cols (rows=C*K*K, cols=N*H*W)
    float* __restrict__ data_im,        // (N, C, H, W)
    int N, int C, int H, int W, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int HW = H * W;
    int cols = N * HW;
    int K2 = K * K;
    int pad = K / 2;

    float sum = 0.0f;
    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            int ih = h + kh - pad;
            int iw = w + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int col = n * HW + ih * W + iw;             // column index
                int row = (c * K2) + kh * K + kw;           // row index
                sum += data_col[row * cols + col];
            }
        }
    }
    data_im[idx] = sum;
}

// Warp-level reduction helper (faster than shared memory)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized warp-level reduction for bias gradient
__global__ void bias_grad_warp_kernel(const float* __restrict__ d_out, float* __restrict__ d_dbias,
                                      int N, int C_out, int H, int W)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int total = N * C_out * H * W;
    
    // Shared memory for per-channel accumulation within block
    extern __shared__ float s_channel_sums[];
    
    // Initialize shared memory (cooperative by all threads)
    for (int i = threadIdx.x; i < C_out; i += blockDim.x) {
        s_channel_sums[i] = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes its element
    if (tid < total) {
        float val = d_out[tid];
        int c = (tid / (H * W)) % C_out;
        
        // Accumulate in shared memory (atomic within block is faster than global atomic)
        atomicAdd(&s_channel_sums[c], val);
    }
    
    __syncthreads();
    
    // Warp-level reduction: reduce shared memory sums within warp, then atomic add to global
    // Each warp processes a subset of channels
    for (int c = warp_id * 32 + lane_id; c < C_out; c += blockDim.x / 32 * 32) {
        float channel_sum = s_channel_sums[c];
        
        // Reduce within warp if multiple warps process same channel range
        // (This is less common, but helps when C_out is small)
        channel_sum = warp_reduce_sum(channel_sum);
        
        // First thread in warp atomic adds to global
        if (lane_id == 0 && channel_sum != 0.0f) {
            atomicAdd(&d_dbias[c], channel_sum);
        }
    }
}

void conv2d_backward_gpu_gemm(
    const float* d_out,
    const float* d_input,
    const float* d_weight,
    float* d_dinput,
    float* d_dweight,
    float* d_dbias,
    int N, int C_in, int H, int W,
    int C_out, int K,
    float* d_im2col,
    cublasHandle_t handle,
    cudaStream_t stream)
{
    int HW = H * W;
    int cols = N * HW;         // columns = spatial*batch
    int rows = C_in * K * K;   // rows of im2col

    // Zero d_dinput (for atomic col2im) and d_dbias
    CUDA_CHECK(cudaMemsetAsync(d_dinput, 0, N * C_in * H * W * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dbias, 0, C_out * sizeof(float), stream));

    // 1) im2col(input) -> d_im2col
    int block = 256;
    int grid = (cols + block - 1) / block;
    im2col_kernel<<<grid, block, 0, stream>>>(d_input, d_im2col, N, C_in, H, W, K);

    // 2) d_weight = d_out * im2col^T
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(handle, stream);
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        C_out,       // m
        rows,        // n
        cols,        // k
        &alpha,
        d_out,       // A: (C_out x cols), lda = C_out (NCHW contiguous per spatial)
        C_out,
        d_im2col,    // B: (rows x cols) column-major, so B^T uses lda = rows
        rows,
        &beta,
        d_dweight,   // C: (C_out x rows)
        C_out);

    // 3) d_input_col = W^T * d_out  -> reuse d_im2col as workspace
    // Interpret weights (row-major C_out x rows) as column-major matrix (rows x C_out) by using opA = N
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        rows,        // m
        cols,        // n
        C_out,       // k
        &alpha,
        d_weight,    // A: viewed as (rows x C_out) column-major
        rows,        // lda = rows (>= rows)
        d_out,       // B: (C_out x cols)
        C_out,
        &beta,
        d_im2col,    // reuse workspace to store d_input_col (rows x cols)
        rows);

    // 4) col2im to accumulate into d_dinput
    int total = N * C_in * H * W;
    grid = (total + block - 1) / block;
    col2im_kernel_noatomic<<<grid, block, 0, stream>>>(d_im2col, d_dinput, N, C_in, H, W, K);

    // 5) bias grad (using warp-level reduction for better performance)
    int total_out = N * C_out * H * W;
    grid = (total_out + block - 1) / block;
    size_t shared_size = C_out * sizeof(float);  // Per-channel accumulation
    bias_grad_warp_kernel<<<grid, block, shared_size, stream>>>(d_out, d_dbias, N, C_out, H, W);

    CUDA_CHECK(cudaGetLastError());
}

// Kernel fallback: dùng bias từ global khi C_out > 512
// Kernel with constant memory bias (faster broadcast access)
__global__ void conv2d_relu_fused_kernel_bias_const(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int bias_offset,  // Offset into constant memory
    float* __restrict__ output,
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
    #pragma unroll
    for (int ci = 0; ci < C_in; ++ci) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int input_idx = ((n * C_in + ci) * H + ih) * W + iw;
                    int weight_idx = ((co * C_in + ci) * K + kh) * K + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    sum += c_bias_all[bias_offset + co];
    sum = fmaxf(sum, 0.0f);
    int out_idx = ((n * C_out + co) * H + h) * W + w;
    output[out_idx] = sum;
}

// Original kernel with global memory bias (kept for fallback)
__global__ void conv2d_relu_fused_kernel_bias(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    #pragma unroll
    for (int ci = 0; ci < C_in; ++ci) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int input_idx = ((n * C_in + ci) * H + ih) * W + iw;
                    int weight_idx = ((co * C_in + ci) * K + kh) * K + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
        sum += bias[co];
    sum = fmaxf(sum, 0.0f);
    int out_idx = ((n * C_out + co) * H + h) * W + w;
    output[out_idx] = sum;
}

// ============================================
// KERNEL SELECTION STRATEGY
// ============================================
// This function implements a multi-level kernel selection strategy:
//
// Level 1: Constant Memory vs Global Memory
//   - Training: use_constant_memory = false → global memory (bias changes every step)
//   - Inference: use_constant_memory = true → constant memory (bias static, faster broadcast)
//   - Rationale: Constant memory provides faster broadcast access but requires D2H copy to update
//
// Level 3: Block Size Tuning
//   - Small HW (<256): 128 threads/block (better occupancy)
//   - Medium HW (<1024): 128 threads/block
//   - Large HW (>=1024): 256 threads/block (better throughput)
//
void conv2d_relu_forward_gpu_fused(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream,
    bool use_constant_memory)
{
    int HW = H * W;
    int block_size = (HW < 256) ? 128 : (HW < 1024) ? 128 : 256;
    
    // Level 1: Determine bias offset for constant memory (if enabled)
    int bias_offset = -1;
    if (use_constant_memory) {
        // Map layer dimensions to constant memory offsets
        if (C_in == 3 && C_out == 256) bias_offset = C_BIAS1_OFFSET;      // Conv1: 3→256
        else if (C_in == 256 && C_out == 128) bias_offset = C_BIAS2_OFFSET;  // Conv2: 256→128
        else if (C_in == 128 && C_out == 128) bias_offset = C_BIAS3_OFFSET;  // Conv3: 128→128
        else if (C_in == 128 && C_out == 256) bias_offset = C_BIAS4_OFFSET;  // Conv4: 128→256
        else if (C_out == 3) bias_offset = C_BIAS5_OFFSET;    // Conv5: 256→3
    }
    
    dim3 block(block_size);
    dim3 grid((HW + block.x - 1) / block.x, C_out, N);
    
    // Use constant memory kernel if bias_offset is valid, otherwise use standard kernel
    if (bias_offset >= 0) {
        // Constant memory kernel: faster broadcast access, no global memory read for bias
        conv2d_relu_fused_kernel_bias_const<<<grid, block, 0, stream>>>(
            d_input, d_weight, bias_offset, d_output,
            N, C_in, H, W, C_out, K);
    } else {
        // Standard kernel: fallback when constant memory is not available
        conv2d_relu_fused_kernel_bias<<<grid, block, 0, stream>>>(
            d_input, d_weight, d_bias, d_output,
            N, C_in, H, W, C_out, K);
    }
    CUDA_CHECK(cudaGetLastError());
}


// Optimized backward kernels
__global__ void conv2d_backward_weight_optimized_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ input,
    float* __restrict__ d_weight,
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    int co = blockIdx.x;
    int ci = blockIdx.y;
    if (co >= C_out || ci >= C_in) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x; // assume 256
    int pad = K / 2;

    // local accumulation per thread
    float local_grad[9] = {0.0f};
    for (int n = 0; n < N; ++n) {
        for (int idx = tid; idx < H * W; idx += block_size) {
            int h = idx / W;
            int w = idx % W;
            float grad_out = d_out[((n * C_out + co) * H + h) * W + w];
            #pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    int ih = h + kh - pad;
                    int iw = w + kw - pad;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float in_val = input[((n * C_in + ci) * H + ih) * W + iw];
                        local_grad[kh * K + kw] += grad_out * in_val;
                    }
                }
            }
        }
    }

    // block reduction per kernel element (no atomic to global)
    __shared__ float s_partial[9][256]; // 9 kernels, up to 256 threads
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        s_partial[k][tid] = local_grad[k];
    }
    __syncthreads();

    // reduction over block_size
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                s_partial[k][tid] += s_partial[k][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            int kh = k / K;
            int kw = k % K;
            int w_idx = ((co * C_in + ci) * K + kh) * K + kw;
            d_weight[w_idx] += s_partial[k][0];
        }
    }
}

__global__ void conv2d_backward_input_optimized_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ weight,
    float* __restrict__ d_input,
    int N, int C_in, int H, int W,
    int C_out, int K)
{
    constexpr int CO_TILE = 8;
    __shared__ float s_weight[CO_TILE][9];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H * W;
    if (idx >= total) return;
    int HW = H * W;
    int n = idx / (C_in * HW);
    int rem = idx % (C_in * HW);
    int ci = rem / HW;
    int rem2 = rem % HW;
    int h = rem2 / W;
    int w = rem2 % W;
    int pad = K / 2;
    float sum = 0.0f;
    int tid = threadIdx.x;
    for (int co_base = 0; co_base < C_out; co_base += CO_TILE) {
        int co_block = min(CO_TILE, C_out - co_base);
        if (tid < co_block * 9) {
            int t = tid;
            int co_local = t / 9;
            int k = t % 9;
            int kh = k / K;
            int kw = k % K;
            int w_idx = ((co_base + co_local) * C_in + ci) * K * K + kh * K + kw;
            s_weight[co_local][k] = weight[w_idx];
        }
        __syncthreads();
        for (int co_local = 0; co_local < co_block; ++co_local) {
            int co = co_base + co_local;
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int h_out = h + kh - pad;  // Flip kernel: +kh instead of -kh
                    int w_out = w + kw - pad;  // Flip kernel: +kw instead of -kw
                    if (h_out >= 0 && h_out < H && w_out >= 0 && w_out < W) {
                        int out_idx = ((n * C_out + co) * H + h_out) * W + w_out;
                        float grad_out = d_out[out_idx];
                        float w_val = s_weight[co_local][kh * K + kw];
                        sum += grad_out * w_val;
                    }
                }
            }
        }
        __syncthreads();
    }
    d_input[idx] = sum;
}

void conv2d_backward_gpu_optimized(
    const float* d_out,
    const float* d_input,
    const float* d_weight,
    float* d_dinput,
    float* d_dweight,
    float* d_dbias,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream)
{
    CUDA_CHECK(cudaMemsetAsync(d_dinput, 0, N * C_in * H * W * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dweight, 0, C_out * C_in * K * K * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dbias, 0, C_out * sizeof(float), stream));

    // Optimized d_input
    int total_in = N * C_in * H * W;
    dim3 block_in(256);
    dim3 grid_in((total_in + block_in.x - 1) / block_in.x);
    conv2d_backward_input_optimized_kernel<<<grid_in, block_in, 0, stream>>>(
        d_out, d_weight, d_dinput,
        N, C_in, H, W, C_out, K);

    // d_weight: Use naive kernel (proven faster for batch size 64 on T4)
    // Optimized kernel has overhead from shared memory reduction that outweighs benefits
    // for small batch sizes and small spatial dimensions (CIFAR-10: 32×32)
    extern __global__ void conv2d_backward_weight_kernel(
        const float*, const float*, float*, int, int, int, int, int, int);
    
    int total_w = C_out * C_in * K * K;
    int block_w = 256;
    int grid_w = (total_w + block_w - 1) / block_w;
    conv2d_backward_weight_kernel<<<grid_w, block_w, 0, stream>>>(
        d_out, d_input, d_dweight,
        N, C_in, H, W, C_out, K);
    
    // Use naive bias kernel (simple and fast)
    extern __global__ void conv2d_backward_bias_kernel(
        const float*, float*, int, int, int, int);

    int block_b = 256;
    int grid_b = (C_out + block_b - 1) / block_b;
    conv2d_backward_bias_kernel<<<grid_b, block_b, 0, stream>>>(
        d_out, d_dbias, N, C_out, H, W);

    CUDA_CHECK(cudaGetLastError());
}

// Optimized pooling with coalesced access
__global__ void maxpool2d_forward_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
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
    
    float m = -1e30f;
    
    #pragma unroll
    for (int dh = 0; dh < 2; ++dh) {
        #pragma unroll
        for (int dw = 0; dw < 2; ++dw) {
            int ih = h_in + dh;
            int iw = w_in + dw;
            
            if (ih < H && iw < W) {
                int in_idx = ((n * C + c) * H + ih) * W + iw;
                float v = input[in_idx];
                if (v > m) m = v;
            }
        }
    }
    
    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = m;
}

void maxpool2d_forward_gpu_optimized(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int H_out = H / 2;
    int W_out = W / 2;
    int HW_out = H_out * W_out;
    
    // Optimized thread block size cho coalesced access
    int block_size = 256;
    dim3 block(block_size);
    dim3 grid((HW_out + block.x - 1) / block.x, C, N);
    
    maxpool2d_forward_optimized_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, N, C, H, W);
    
    CUDA_CHECK(cudaGetLastError());
}

// Optimized upsampling
__global__ void upsample2d_forward_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int n = blockIdx.z;
    int c = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int H_out = H * 2;
    int W_out = W * 2;
    int HW_out = H_out * W_out;
    
    if (idx >= HW_out) return;
    
    int h_out = idx / W_out;
    int w_out = idx % W_out;
    int h_in = h_out / 2;
    int w_in = w_out / 2;
    
    // Nearest neighbor upsampling: copy từ input
    int in_idx = ((n * C + c) * H + h_in) * W + w_in;
    float val = input[in_idx];
    
    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = val;
}

void upsample2d_forward_gpu_optimized(
    const float* d_input,
    float* d_output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int H_out = H * 2;
    int W_out = W * 2;
    int HW_out = H_out * W_out;
    
    // Optimized thread block size
    int block_size = 256;
    dim3 block(block_size);
    dim3 grid((HW_out + block.x - 1) / block.x, C, N);
    
    upsample2d_forward_optimized_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, N, C, H, W);

    CUDA_CHECK(cudaGetLastError());
}


// Helper function to update constant memory biases
void update_constant_memory_biases(
    const float* d_b1, int c1,
    const float* d_b2, int c2,
    const float* d_b3, int c3,
    const float* d_b4, int c4,
    const float* d_b5, int c5,
    cudaStream_t stream)
{
    std::vector<float> h_bias_all(256 + 128 + 128 + 256 + 3);
    CUDA_CHECK(cudaMemcpyAsync(h_bias_all.data() + 0, d_b1, c1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_bias_all.data() + 256, d_b2, c2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_bias_all.data() + 384, d_b3, c3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_bias_all.data() + 512, d_b4, c4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_bias_all.data() + 768, d_b5, c5 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bias_all, h_bias_all.data(), h_bias_all.size() * sizeof(float)));
}

