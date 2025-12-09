#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "layers_gpu.h"

int main() {
    std::cout << "Running GPU layer tests...\n";

    // -------- Test Conv2D + ReLU --------
    int N = 1, C_in = 1, H = 4, W = 4;
    int C_out = 1, K = 3;

    std::vector<float> h_input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16
    };

    std::vector<float> h_weight(9, 1.0f);
    std::vector<float> h_bias(1, 0.0f);

    std::vector<float> h_conv(H*W);
    std::vector<float> h_relu(H*W);

    float *d_in, *d_w, *d_b, *d_conv, *d_relu;
    cudaMalloc(&d_in,   h_input.size()*sizeof(float));
    cudaMalloc(&d_w,    h_weight.size()*sizeof(float));
    cudaMalloc(&d_b,    h_bias.size()*sizeof(float));
    cudaMalloc(&d_conv, h_conv.size()*sizeof(float));
    cudaMalloc(&d_relu, h_relu.size()*sizeof(float));

    cudaMemcpy(d_in, h_input.data(), h_input.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,  h_weight.data(), h_weight.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,  h_bias.data(), h_bias.size()*sizeof(float), cudaMemcpyHostToDevice);

    conv2d_forward_gpu_naive(d_in, d_w, d_b, d_conv, N, C_in, H, W, C_out, K);
    relu_forward_gpu(d_conv, d_relu, H*W);

    cudaMemcpy(h_conv.data(), d_conv, h_conv.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu.data(), d_relu, h_relu.size()*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Conv output:\n";
    for (auto v : h_conv) std::cout << v << " ";
    std::cout << "\nReLU output:\n";
    for (auto v : h_relu) std::cout << v << " ";
    std::cout << "\n";

    // -------- Test MaxPool 2x2 forward --------
    std::cout << "\nTesting MaxPool 2x2 forward...\n";

    int C = 1;
    int H_out = H / 2;
    int W_out = W / 2;

    std::vector<float> h_pool(H_out * W_out);
    float *d_pool_in, *d_pool_out;

    cudaMalloc(&d_pool_in,  h_input.size()*sizeof(float));
    cudaMalloc(&d_pool_out, h_pool.size()*sizeof(float));

    cudaMemcpy(d_pool_in, h_input.data(), h_input.size()*sizeof(float), cudaMemcpyHostToDevice);
    maxpool2d_forward_gpu(d_pool_in, d_pool_out, N, C, H, W);
    cudaMemcpy(h_pool.data(), d_pool_out, h_pool.size()*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "MaxPool output (2x2):\n";
    for (int i = 0; i < H_out; ++i) {
        for (int j = 0; j < W_out; ++j) {
            std::cout << h_pool[i * W_out + j] << " ";
        }
        std::cout << "\n";
    }

    // -------- Test Upsample 2x forward --------
    std::cout << "\nTesting Upsample 2x (nearest)...\n";

    int H_u = 2, W_u = 2;
    std::vector<float> h_small = {
        1, 2,
        3, 4
    };

    int H_u_out = H_u * 2;
    int W_u_out = W_u * 2;
    std::vector<float> h_up(H_u_out * W_u_out);

    float *d_up_in, *d_up_out;
    cudaMalloc(&d_up_in,  h_small.size()*sizeof(float));
    cudaMalloc(&d_up_out, h_up.size()*sizeof(float));

    cudaMemcpy(d_up_in, h_small.data(), h_small.size()*sizeof(float), cudaMemcpyHostToDevice);
    upsample2d_forward_gpu(d_up_in, d_up_out, N, C, H_u, W_u);
    cudaMemcpy(h_up.data(), d_up_out, h_up.size()*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Upsampled output (4x4):\n";
    for (int i = 0; i < H_u_out; ++i) {
        for (int j = 0; j < W_u_out; ++j) {
            std::cout << h_up[i * W_u_out + j] << " ";
        }
        std::cout << "\n";
    }

    // -------- Test MSE Loss --------
    std::cout << "\nTesting MSE Loss (forward + backward)...\n";

    std::vector<float> h_pred   = {1, 2, 3, 4};
    std::vector<float> h_target = {1, 1, 1, 1};
    int total = h_pred.size();

    float *d_pred, *d_target, *d_dpred;
    cudaMalloc(&d_pred,   total * sizeof(float));
    cudaMalloc(&d_target, total * sizeof(float));
    cudaMalloc(&d_dpred,  total * sizeof(float));

    cudaMemcpy(d_pred,   h_pred.data(),   total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    float loss = mse_loss_forward_gpu(d_pred, d_target, total);
    std::cout << "MSE loss (GPU) = " << loss << " (expected ~3.5)\n";

    mse_loss_backward_gpu(d_pred, d_target, d_dpred, total);
    std::vector<float> h_dpred(total);
    cudaMemcpy(h_dpred.data(), d_dpred, total * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "MSE dPred (GPU):\n";
    for (int i = 0; i < total; ++i) {
        std::cout << h_dpred[i] << " ";
    }
    std::cout << "\nExpected (2*(pred-target)/N):\n";
    for (int i = 0; i < total; ++i) {
        float diff = h_pred[i] - h_target[i];
        std::cout << 2.0f * diff / float(total) << " ";
    }
    std::cout << "\n";

    // -------- Test ReLU backward --------
    std::cout << "\nTesting ReLU backward...\n";

    std::vector<float> h_x = {-1.0f, 2.0f, -3.0f, 4.0f};
    std::vector<float> h_dout = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> h_dx(4, 0.0f);

    float *d_x, *d_dout2, *d_dx;
    cudaMalloc(&d_x,      4 * sizeof(float));
    cudaMalloc(&d_dout2,  4 * sizeof(float));
    cudaMalloc(&d_dx,     4 * sizeof(float));

    cudaMemcpy(d_x,      h_x.data(),    4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout2,  h_dout.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);

    relu_backward_gpu(d_dout2, d_x, d_dx, 4);

    cudaMemcpy(h_dx.data(), d_dx, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "ReLU backward dX (GPU):\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << h_dx[i] << " ";
    }
    std::cout << "\nExpected (0, 1, 0, 1):\n0 1 0 1\n";

    // -------- Test Upsample backward --------
    std::cout << "\nTesting Upsample 2x backward...\n";

    int H_b = 2, W_b = 2;
    int H_b_out = H_b * 2;
    int W_b_out = W_b * 2;

    std::vector<float> h_dy(H_b_out * W_b_out, 1.0f);  // mọi gradient = 1
    std::vector<float> h_dx_up(H_b * W_b, 0.0f);

    float *d_dy_up, *d_dx_up2;
    cudaMalloc(&d_dy_up,  h_dy.size()    * sizeof(float));
    cudaMalloc(&d_dx_up2, h_dx_up.size() * sizeof(float));

    cudaMemcpy(d_dy_up, h_dy.data(), h_dy.size()*sizeof(float), cudaMemcpyHostToDevice);

    upsample2d_backward_gpu(d_dy_up, d_dx_up2, 1, 1, H_b, W_b);

    cudaMemcpy(h_dx_up.data(), d_dx_up2, h_dx_up.size()*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Upsample backward dX (GPU):\n";
    for (int i = 0; i < H_b * W_b; ++i) {
        std::cout << h_dx_up[i] << " ";
    }
    std::cout << "\nExpected (4, 4, 4, 4):\n4 4 4 4\n";

    // -------- Test MaxPool backward --------
    std::cout << "\nTesting MaxPool 2x2 backward...\n";

    // Input 2x2: [1,2; 3,4] -> max = 4 (bottom-right)
    std::vector<float> h_pool_in = {
        1, 2,
        3, 4
    };
    std::vector<float> h_pool_dout(1, 1.0f);  // gradient ở output = 1
    std::vector<float> h_pool_dx(4, 0.0f);

    float *d_pool_in2, *d_pool_dout2, *d_pool_dx2;
    cudaMalloc(&d_pool_in2,   4 * sizeof(float));
    cudaMalloc(&d_pool_dout2, 1 * sizeof(float));
    cudaMalloc(&d_pool_dx2,   4 * sizeof(float));

    cudaMemcpy(d_pool_in2,   h_pool_in.data(),   4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pool_dout2, h_pool_dout.data(), 1 * sizeof(float), cudaMemcpyHostToDevice);

    maxpool2d_backward_gpu(d_pool_dout2, d_pool_in2, d_pool_dx2, 1, 1, 2, 2);

    cudaMemcpy(h_pool_dx.data(), d_pool_dx2, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "MaxPool backward dX (GPU):\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << h_pool_dx[i] << " ";
    }
    std::cout << "\nExpected (0, 0, 0, 1):\n0 0 0 1\n";

        // -------- Test Conv2D backward (K=1) --------
    std::cout << "\nTesting Conv2D backward (K=1)...\n";

    int Nb = 1, C_in_b = 1, Hb = 2, Wb = 2, C_out_b = 1, Kb = 1;

    std::vector<float> h_xb = {
        1, 2,
        3, 4
    };
    std::vector<float> h_wb = {1.0f};
    std::vector<float> h_bb = {0.0f};

    std::vector<float> h_yb(Hb * Wb);
    std::vector<float> h_dyb(Hb * Wb, 1.0f); // d_out = 1 hết
    std::vector<float> h_dxb(Hb * Wb);
    std::vector<float> h_dwb(1);
    std::vector<float> h_dbb(1);

    float *d_xb, *d_wb, *d_bb, *d_yb, *d_dyb, *d_dxb, *d_dwb, *d_dbb;
    cudaMalloc(&d_xb,  h_xb.size() * sizeof(float));
    cudaMalloc(&d_wb,  h_wb.size() * sizeof(float));
    cudaMalloc(&d_bb,  h_bb.size() * sizeof(float));
    cudaMalloc(&d_yb,  h_yb.size() * sizeof(float));
    cudaMalloc(&d_dyb, h_dyb.size() * sizeof(float));
    cudaMalloc(&d_dxb, h_dxb.size() * sizeof(float));
    cudaMalloc(&d_dwb, h_dwb.size() * sizeof(float));
    cudaMalloc(&d_dbb, h_dbb.size() * sizeof(float));

    cudaMemcpy(d_xb,  h_xb.data(),  h_xb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wb,  h_wb.data(),  h_wb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bb,  h_bb.data(),  h_bb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dyb, h_dyb.data(), h_dyb.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Forward (để chắc chắn conv không lỗi, nhưng thực ra test backward không bắt buộc)
    conv2d_forward_gpu_naive(d_xb, d_wb, d_bb, d_yb,
                             Nb, C_in_b, Hb, Wb,
                             C_out_b, Kb);

    // Backward
    conv2d_backward_gpu_naive(
        d_dyb,   // d_out
        d_xb,    // input X
        d_wb,    // weight W
        d_dxb,   // dX
        d_dwb,   // dW
        d_dbb,   // dB
        Nb, C_in_b, Hb, Wb,
        C_out_b, Kb);

    cudaMemcpy(h_dxb.data(), d_dxb, h_dxb.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dwb.data(), d_dwb, h_dwb.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dbb.data(), d_dbb, h_dbb.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "dX (GPU):\n";
    for (int i = 0; i < Hb * Wb; ++i) {
        std::cout << h_dxb[i] << " ";
    }
    std::cout << "\nExpected dX: 1 1 1 1\n";

    std::cout << "dW (GPU): " << h_dwb[0] << "\n";
    std::cout << "Expected dW: 10\n";

    std::cout << "dB (GPU): " << h_dbb[0] << "\n";
    std::cout << "Expected dB: 4\n";


    return 0;
}