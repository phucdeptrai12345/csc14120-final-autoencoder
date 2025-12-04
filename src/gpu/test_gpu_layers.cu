#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "layers_gpu.h"

int main() {
    std::cout << "Running Conv2D + ReLU GPU test...\n";

    int N = 1, C_in = 1, H = 4, W = 4;
    int C_out = 1, K = 3;

    // host input 1..16
    std::vector<float> h_input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16
    };

    std::vector<float> h_weight(9, 1.0f); // kernel all 1
    std::vector<float> h_bias(1, 0.0f);

    std::vector<float> h_output(H*W);
    std::vector<float> h_relu(H*W);

    float *d_in, *d_w, *d_b, *d_out, *d_relu;
    cudaMalloc(&d_in, h_input.size()*4);
    cudaMalloc(&d_w, 9*4);
    cudaMalloc(&d_b, 4);
    cudaMalloc(&d_out, h_output.size()*4);
    cudaMalloc(&d_relu, h_relu.size()*4);

    cudaMemcpy(d_in, h_input.data(), h_input.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_weight.data(), 9*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_bias.data(), 4, cudaMemcpyHostToDevice);

    conv2d_forward_gpu_naive(d_in, d_w, d_b, d_out, N, C_in, H, W, C_out, K);
    relu_forward_gpu(d_out, d_relu, H*W);

    cudaMemcpy(h_output.data(), d_out, h_output.size()*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu.data(), d_relu, h_relu.size()*4, cudaMemcpyDeviceToHost);

    std::cout << "Conv output:" << std::endl;
    for (auto v : h_output) std::cout << v << " ";
    std::cout << "\nReLU output:" << std::endl;
    for (auto v : h_relu) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
