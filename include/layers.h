#ifndef LAYERS_H
#define LAYERS_H

#include "utils.h"

// --- Convolutional Layer ---
class Conv2D
{
public:
    int in_channels, out_channels;
    int kernel_size; // Giả sử kernel vuông (3x3)
    int stride, padding;
    
    std::vector<float> weights;
    std::vector<float> biases;

    // Gradient
    std::vector<float> grad_weights;
    std::vector<float> grad_biases;

    // Save Input
    Tensor input_cache;

    Conv2D(int in_c, int out_c, int k_size, int s = 1, int p = 1);

    // forward propagation
    void forward(const Tensor &input, Tensor &output);

    // back propagation
    void backward(const Tensor &dout, Tensor &din, float lr);
};

// --- ReLU Activation ---
class ReLU
{
public:
    Tensor input_cache; // Cần lưu input để biết vị trí nào > 0

    void forward(const Tensor &input, Tensor &output);
    void backward(const Tensor &dout, Tensor &din);
};

// --- Max Pooling Layer ---
class MaxPool2D
{
public:
    int pool_size, stride;

    // Cần lưu chỉ số (index) của vị trí max để backprop
    // mask lưu vị trí: mask(c, h, w) = index trong input
    Tensor index_mask;
    Tensor input_shape_cache; // Lưu kích thước input gốc

    MaxPool2D(int size = 2, int s = 2);

    void forward(const Tensor &input, Tensor &output);
    void backward(const Tensor &dout, Tensor &din);
};

// --- UpSampling Layer ---
class UpSample2D
{
public:
    int scale_factor;

    UpSample2D(int scale = 2);

    void forward(const Tensor &input, Tensor &output);
    void backward(const Tensor &dout, Tensor &din);
};

#endif // LAYERS_H