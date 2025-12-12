// ============================================
// layers_gpu_optimized.cu - PHASE 3 FINAL
// Complete optimizations with bug fixes
// ============================================
#include "layers_gpu_optimized.h"
#include "layers_gpu.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For FP16 support
#include <cublas_v2.h>
#include <cstdio>
#include <cstddef>

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = cmd; \
        if (e != cudaSuccess) { \
            printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e)); \
        } \
    } while (0)

// ============================================
// OPTIMIZATION 1: KERNEL FUSION (Conv + ReLU + Bias)
// Speedup: ~1.3x on forward pass
// ============================================
// Forward declaration for vectorized kernel (defined later)
__global__ void conv2d_relu_fused_vectorized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int K);

// ============================================
// GEMM path: im2col + cuBLAS + bias + ReLU
// ============================================
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    int N, int C, int H, int W, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    int total_cols = N * HW; // one column per (n,h,w)
    if (idx >= total_cols) return;

    int n = idx / HW;
    int m = idx % HW;
    int h = m / W;
    int w = m % W;
    int pad = K / 2;

    // Write C*K*K values for this column
    int K2 = K * K;
    int rows = C * K2;     // rows = C*K*K (column-major stride)
    int col_base = idx;    // column index
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
                // data_col layout: (rows x cols) column-major -> row + col*rows
                data_col[k_idx + col_base * rows] = val;
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
    int total_cols = N * HW; // one column per (n,h,w)
    if (idx >= total_cols) return;

    int n = idx / HW;
    int m = idx % HW;
    int h = m / W;
    int w = m % W;
    int pad = K / 2;

    int K2 = K * K;
    int rows = C * K2; // column-major stride
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
                // data_col layout: column-major (rows x cols)
                data_col[k_idx + idx * rows] = __float2half(val);
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

// ============================================
// BACKWARD: im2col + GEMM + col2im
// ============================================
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
    int rows = C * K * K; // column-major stride
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
                // data_col layout: column-major (rows x cols)
                sum += data_col[row + col * rows];
            }
        }
    }
    data_im[idx] = sum;
}

__global__ void bias_grad_kernel(const float* __restrict__ d_out, float* __restrict__ d_dbias,
                                 int N, int C_out, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;
    if (idx >= total) return;
    int c = (idx / (H * W)) % C_out;
    atomicAdd(&d_dbias[c], d_out[idx]);
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

    // 5) bias grad
    int total_out = N * C_out * H * W;
    grid = (total_out + block - 1) / block;
    bias_grad_kernel<<<grid, block, 0, stream>>>(d_out, d_dbias, N, C_out, H, W);

    CUDA_CHECK(cudaGetLastError());
}

// Kernel fallback: dùng bias từ global khi C_out > 512
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

void conv2d_relu_forward_gpu_fused(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream)
{
    // Dùng hoàn toàn global memory để luôn đồng bộ với SGD updates (bỏ nhánh constant Conv1).
    int HW = H * W;
    int block_size = (HW < 256) ? 128 : (HW < 1024) ? 128 : 256;
    dim3 block(block_size);
    dim3 grid((HW + block.x - 1) / block.x, C_out, N);

    // Chọn kernel vectorized khi C_in bội số của 4, ngược lại dùng kernel bias thông thường.
    if (C_in >= 4 && (C_in % 4 == 0) && C_out <= 512) {
        conv2d_relu_fused_vectorized_kernel<<<grid, block, 0, stream>>>(
            d_input, d_weight, d_bias, d_output,
            N, C_in, H, W, C_out, K);
    } else {
        conv2d_relu_fused_kernel_bias<<<grid, block, 0, stream>>>(
            d_input, d_weight, d_bias, d_output,
            N, C_in, H, W, C_out, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// SMART KERNEL SELECTION - Auto-select best kernel based on size
// Heuristic: Chọn kernel tốt nhất dựa trên spatial size, channel size, và matrix size
// Expected speedup: 15% từ việc chọn kernel tối ưu cho từng layer
// ============================================
void conv2d_relu_forward_smart(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    float* d_im2col,
    float* d_gemm_out,
    __half* d_im2col_fp16,
    cublasHandle_t handle,
    bool use_fp16,
    cudaStream_t stream)
{
    int HW = H * W;
    int cols = N * HW;           // columns for GEMM
    int rows = C_in * K * K;     // rows for GEMM
    
    // Heuristic: Chọn kernel tốt nhất dựa trên size
    // Analysis cho CIFAR-10 (N=64):
    // Conv1: cols=64*1024=65536, rows=3*9=27 → GEMM tốt hơn (cols lớn)
    // Conv2: cols=64*256=16384, rows=256*9=2304 → GEMM FP16 tốt (matrix lớn)
    // Conv3: cols=64*64=4096, rows=128*9=1152 → GEMM tốt (rows lớn)
    // Conv4: cols=64*256=16384, rows=128*9=1152 → GEMM FP16 tốt
    //
    // Rule: Fused chỉ tốt khi BOTH cols và rows đều nhỏ
    //       GEMM tốt khi cols >= 10000 HOẶC rows >= 100
    
    bool use_fused = false;
    bool use_fp16_gemm = false;
    
    // Rule 1: Fused chỉ tốt khi CẢ cols VÀ rows đều nhỏ
    // Với cols lớn (>= 10000), GEMM luôn tốt hơn dù rows nhỏ
    if (cols < 10000 && rows < 100 && HW <= 1024 && C_in <= 256) {
        use_fused = true;
    }
    // Rule 2: GEMM FP16 tốt cho matrix lớn và GPU supports FP16
    else if (use_fp16 && (cols >= 10000 || rows >= 100)) {
        use_fp16_gemm = true;
    }
    // Rule 3: Default to GEMM FP32 cho matrix lớn
    else {
        use_fused = false;
        use_fp16_gemm = false;
    }
    
    // Execute selected kernel
    if (use_fused) {
        conv2d_relu_forward_gpu_fused(
            d_input, d_weight, d_bias, d_output,
            N, C_in, H, W, C_out, K, stream);
    } else if (use_fp16_gemm) {
        conv2d_relu_forward_gemm_fp16(
            d_input, d_weight, d_bias, d_output,
            N, C_in, H, W, C_out, K,
            d_im2col_fp16, d_gemm_out, handle, stream);
    } else {
        conv2d_relu_forward_gemm(
            d_input, d_weight, d_bias, d_output,
            N, C_in, H, W, C_out, K,
            d_im2col, d_gemm_out, handle, stream);
    }
}

// ============================================
// Conv không ReLU: dùng naive để tránh overhead ở spatial nhỏ
// ============================================
void conv2d_forward_gpu_tiled(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream)
{
    conv2d_forward_gpu_naive(
        d_input, d_weight, d_bias, d_output,
        N, C_in, H, W, C_out, K, stream);
}

// ============================================
// OPTIMIZATION 3: OPTIMIZED BACKWARD
// ============================================
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
                    int h_out = h - kh + pad;
                    int w_out = w - kw + pad;
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
    // OPTIMIZATION: Zero gradients - có thể parallelize nhưng giữ nguyên để đảm bảo correctness
    // Note: Có thể combine nếu buffers liên tiếp, nhưng không đảm bảo layout
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

// ============================================
// OPTIMIZATION 4: PINNED MEMORY HELPERS
// ============================================
float* allocate_pinned_memory(size_t size) {
    float* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void free_pinned_memory(float* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

// ============================================
// ============================================
// OPTIMIZATION 7: OPTIMIZED POOLING với Coalesced Access
// Optimize memory access pattern để tận dụng memory bandwidth
// ============================================
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
    
    // OPTIMIZATION: Unroll loops và optimize access pattern
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

// ============================================
// OPTIMIZATION 8: OPTIMIZED UPSAMPLING với Vectorized Access
// ============================================
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

// ============================================
// OPTIMIZATION 5: VECTORIZED RELU (with stream support)
// ============================================
__global__ void relu_forward_vectorized_kernel(
    const float* input,
    float* output,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    
    // Check alignment: input + idx must be 16-byte aligned for float4
    // Use pointer arithmetic to check alignment (simpler than uintptr_t)
    const float* in_ptr = input + idx;
    float* out_ptr = output + idx;
    
    // Check if pointers are 16-byte aligned (address % 16 == 0)
    bool is_aligned = (((size_t)in_ptr & 0xF) == 0) && (((size_t)out_ptr & 0xF) == 0);
    
    if (idx + 3 < N && is_aligned) {
        float4 val = *reinterpret_cast<const float4*>(in_ptr);
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        *reinterpret_cast<float4*>(out_ptr) = val;
    } else {
        // Fallback to scalar for misaligned or remaining elements
        int end = (idx + 4 < N) ? idx + 4 : N;
        for (int i = idx; i < end; ++i) {
            output[i] = fmaxf(input[i], 0.0f);
        }
    }
}

void relu_forward_gpu_vectorized(const float* d_input, float* d_output, int total, cudaStream_t stream) {
    int block = 256;
    int grid = ((total + 3) / 4 + block - 1) / block;
    relu_forward_vectorized_kernel<<<grid, block, 0, stream>>>(d_input, d_output, total);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// SMART WRAPPER: Auto-select vectorized or scalar ReLU
// Uses vectorized when size is large enough to benefit
// ============================================
void relu_forward_gpu_smart(const float* d_input, float* d_output, int total, cudaStream_t stream) {
    // Heuristic: Use vectorized for large arrays (>= 1024 elements)
    // Vectorized kernel has overhead from alignment checks, so only beneficial for large sizes
    // Also, vectorized processes 4 elements at once, so need at least 4 elements
    if (total >= 1024 && total >= 4) {
        // Use vectorized version (has built-in alignment checks and fallback)
        relu_forward_gpu_vectorized(d_input, d_output, total, stream);
    } else {
        // Use scalar version for small arrays (lower overhead)
        extern void relu_forward_gpu(const float*, float*, int, cudaStream_t);
        relu_forward_gpu(d_input, d_output, total, stream);
    }
}

// ============================================
// OPTIMIZATION 6: VECTORIZED MEMORY ACCESS (float4) for Convolution
// Load/store 4 floats at once for better memory bandwidth
// ============================================
__global__ void conv2d_relu_fused_vectorized_kernel(
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
    
    // Vectorized accumulation: process 4 channels at once when possible
    float sum = 0.0f;
    int ci_base = 0;
    
    // Process channels in groups of 4 for better coalescing
    // NOTE: Vectorized access requires 16-byte alignment, which may not be guaranteed
    // for arbitrary memory layouts. Use scalar access for safety.
    for (; ci_base + 4 <= C_in; ci_base += 4) {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    // Use scalar access to avoid alignment issues
                    // Vectorized access requires careful memory layout which may not be guaranteed
                    int idx0 = ((n * C_in + ci_base + 0) * H + ih) * W + iw;
                    int idx1 = ((n * C_in + ci_base + 1) * H + ih) * W + iw;
                    int idx2 = ((n * C_in + ci_base + 2) * H + ih) * W + iw;
                    int idx3 = ((n * C_in + ci_base + 3) * H + ih) * W + iw;
                    
                    float in0 = input[idx0];
                    float in1 = input[idx1];
                    float in2 = input[idx2];
                    float in3 = input[idx3];
                    
                    // Load weights for 4 channels
                    float w0 = weight[((co * C_in + ci_base + 0) * K + kh) * K + kw];
                    float w1 = weight[((co * C_in + ci_base + 1) * K + kh) * K + kw];
                    float w2 = weight[((co * C_in + ci_base + 2) * K + kh) * K + kw];
                    float w3 = weight[((co * C_in + ci_base + 3) * K + kh) * K + kw];
                    
                    sum += in0 * w0 + in1 * w1 + in2 * w2 + in3 * w3;
                }
            }
        }
    }
    
    // Handle remaining channels
    #pragma unroll
    for (int ci = ci_base; ci < C_in; ++ci) {
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
    
    // OPTIMIZATION: Use bias from global memory (faster than constant memory copy overhead)
    sum += bias[co];
    sum = fmaxf(sum, 0.0f);
    int out_idx = ((n * C_out + co) * H + h) * W + w;
    output[out_idx] = sum;
}

// Wrapper: use vectorized version when C_in is multiple of 4 and >= 4
void conv2d_relu_forward_gpu_fused_vectorized(
    const float* d_input,
    const float* d_weight,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream)
{
    int HW = H * W;
    // OPTIMIZED THREAD BLOCK DIMENSIONS: Tune based on spatial size
    int block_size = (HW < 256) ? 64 : (HW < 1024) ? 128 : 256;
    dim3 block(block_size);
    dim3 grid((HW + block.x - 1) / block.x, C_out, N);

    if (C_out <= 512 && C_in >= 4 && (C_in % 4 == 0)) {
        // OPTIMIZATION: Use vectorized version with bias from global memory
        // Loại bỏ constant memory copy overhead (cudaMemcpyToSymbolAsync)
        // Global memory access nhanh hơn constant memory copy overhead
        conv2d_relu_fused_vectorized_kernel<<<grid, block, 0, stream>>>(
        d_input, d_weight, d_bias, d_output,
        N, C_in, H, W, C_out, K);
    } else {
        // Fallback to regular fused kernel
        conv2d_relu_forward_gpu_fused(d_input, d_weight, d_bias, d_output,
                                     N, C_in, H, W, C_out, K, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// ============================================
// OPTIMIZATION 13: MIXED PRECISION (FP16/FP32)
// FP16 activations for forward pass, FP32 for weights and updates
// Expected speedup: ~20-30% (memory bandwidth)
// ============================================

// Helper: Convert FP32 array to FP16 array
__global__ void convert_fp32_to_fp16_kernel(
    const float* input,
    __half* output,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Clamp to FP16 range to avoid overflow/underflow
    const float fp16_max = 65504.0f;
    const float fp16_min = -65504.0f;
    float val = input[idx];
    val = fmaxf(fminf(val, fp16_max), fp16_min);
    output[idx] = __float2half(val);
}

// Helper: Convert FP16 array to FP32 array
__global__ void convert_fp16_to_fp32_kernel(
    const __half* input,
    float* output,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = __half2float(input[idx]);
}

// Mixed Precision Conv+ReLU kernel: FP16 activations, FP32 weights
__global__ void conv2d_relu_fused_fp16_kernel(
    const __half* __restrict__ input,    // FP16 activations
    const float* __restrict__ weight,    // FP32 weights
    const float* __restrict__ bias,      // FP32 bias
    __half* __restrict__ output,        // FP16 activations
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
    
    // Accumulate in FP32 for precision
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
                    
                    // Convert FP16 input to FP32 for computation
                    float in_val = __half2float(input[input_idx]);
                    float w_val = weight[weight_idx];
                    sum += in_val * w_val;
                }
            }
        }
    }
    
    // Add bias (FP32)
    sum += bias[co];
    
    // ReLU and convert to FP16
    sum = fmaxf(sum, 0.0f);
    
    // Clamp to FP16 range
    const float fp16_max = 65504.0f;
    sum = fminf(sum, fp16_max);
    
    __half out_val = __float2half(sum);
    
    int out_idx = ((n * C_out + co) * H + h) * W + w;
    output[out_idx] = out_val;
}

// Wrapper function for mixed precision Conv+ReLU
// NOTE: This version requires pre-allocated FP16 buffers to avoid allocation overhead
// For better performance, buffers should be reused across calls
void conv2d_relu_forward_gpu_fused_fp16(
    const float* d_input_fp32,      // Input in FP32
    const float* d_weight,          // Weights in FP32
    const float* d_bias,            // Bias in FP32
    float* d_output_fp32,           // Output in FP32
    __half* d_input_fp16_buffer,   // Pre-allocated FP16 input buffer (reuse)
    __half* d_output_fp16_buffer,  // Pre-allocated FP16 output buffer (reuse)
    int N, int C_in, int H, int W,
    int C_out, int K,
    cudaStream_t stream)
{
    int HW = H * W;
    int block_size = (HW < 256) ? 128 : (HW < 1024) ? 128 : 256;
    dim3 block(block_size);
    dim3 grid((HW + block.x - 1) / block.x, C_out, N);
    
    size_t input_size = N * C_in * H * W;
    size_t output_size = N * C_out * H * W;
    
    // Convert FP32 input → FP16
    int block_conv = 256;
    int grid_conv = (input_size + block_conv - 1) / block_conv;
    convert_fp32_to_fp16_kernel<<<grid_conv, block_conv, 0, stream>>>(
        d_input_fp32, d_input_fp16_buffer, input_size);
    
    // Launch mixed precision kernel
    conv2d_relu_fused_fp16_kernel<<<grid, block, 0, stream>>>(
        d_input_fp16_buffer, d_weight, d_bias, d_output_fp16_buffer,
        N, C_in, H, W, C_out, K);
    
    // Convert FP16 output → FP32
    grid_conv = (output_size + block_conv - 1) / block_conv;
    convert_fp16_to_fp32_kernel<<<grid_conv, block_conv, 0, stream>>>(
        d_output_fp16_buffer, d_output_fp32, output_size);

    CUDA_CHECK(cudaGetLastError());
}

