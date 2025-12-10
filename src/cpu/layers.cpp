#include "../../include/layers.h"
#include <cmath>
#include <algorithm>

// ================== Conv2D ==================

Conv2D::Conv2D(int in_c, int out_c, int k_size, int s, int p)
    : in_channels(in_c), out_channels(out_c), kernel_size(k_size), stride(s), padding(p)
{

    // Init weights and biases
    weights.resize(out_channels * in_channels * kernel_size * kernel_size);
    biases.resize(out_channels, 0.0f);

    grad_weights.resize(weights.size(), 0.0f);
    grad_biases.resize(biases.size(), 0.0f);

    initialize_weights(weights, weights.size());
}

void Conv2D::forward(const Tensor &input, Tensor &output)
{
    this->input_cache = input; // Lưu input để dùng cho backward

    // output size
    // H_out = (H_in + 2*pad - kernel) / stride + 1
    int out_h = (input.h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (input.w + 2 * padding - kernel_size) / stride + 1;

    // Reset output data
    if (output.data.empty() || output.c != out_channels || output.h != out_h)
    {
        output = Tensor(out_channels, out_h, out_w);
    }
    else
    {
        output.zero_grad(); 
    }

    // Naive Convolution: 6 nested loops
    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int oh = 0; oh < out_h; ++oh)
        {
            for (int ow = 0; ow < out_w; ++ow)
            {
                // Calculate Value
                float sum = biases[oc];

                for (int ic = 0; ic < in_channels; ++ic)
                {
                    for (int kh = 0; kh < kernel_size; ++kh)
                    {
                        for (int kw = 0; kw < kernel_size; ++kw)
                        {

                            int in_row = oh * stride + kh - padding;
                            int in_col = ow * stride + kw - padding;

                            if (in_row >= 0 && in_row < input.h && in_col >= 0 && in_col < input.w)
                            {
                                // Index weights
                                int w_idx = oc * (in_channels * kernel_size * kernel_size) +
                                            ic * (kernel_size * kernel_size) +
                                            kh * kernel_size + kw;
                                sum += input(ic, in_row, in_col) * weights[w_idx];
                            }
                        }
                    }
                    
                }
                output(oc, oh, ow) = sum;
            }
        }
    }
}

void Conv2D::backward(const Tensor &dout, Tensor &din, float lr)
{
    // Reset gradient input
    din = Tensor(in_channels, input_cache.h, input_cache.w);

    std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
    std::fill(grad_biases.begin(), grad_biases.end(), 0.0f);

    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int oh = 0; oh < dout.h; ++oh)
        {
            for (int ow = 0; ow < dout.w; ++ow)
            {

                float d_val = dout(oc, oh, ow);
                grad_biases[oc] += d_val; // Gradient of bias

                for (int ic = 0; ic < in_channels; ++ic)
                {
                    for (int kh = 0; kh < kernel_size; ++kh)
                    {
                        for (int kw = 0; kw < kernel_size; ++kw)
                        {

                            int in_row = oh * stride + kh - padding;
                            int in_col = ow * stride + kw - padding;

                            if (in_row >= 0 && in_row < input_cache.h && in_col >= 0 && in_col < input_cache.w)
                            {
                                int w_idx = oc * (in_channels * kernel_size * kernel_size) +
                                            ic * (kernel_size * kernel_size) +
                                            kh * kernel_size + kw;

                                // Gradient w.r.t weights: input * dout
                                grad_weights[w_idx] += input_cache(ic, in_row, in_col) * d_val;

                                // Gradient w.r.t input: weight * dout (Backprop lỗi về trước)
                                din(ic, in_row, in_col) += weights[w_idx] * d_val;
                            }
                        }
                    }
                }
            }
        }
    }

    // update weights an biases
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] -= lr * grad_weights[i];
    }
    for (size_t i = 0; i < biases.size(); ++i)
    {
        biases[i] -= lr * grad_biases[i];
    }
}

// ================== ReLU ==================

void ReLU::forward(const Tensor &input, Tensor &output)
{
    input_cache = input;
    output = Tensor(input.c, input.h, input.w);
    for (size_t i = 0; i < input.data.size(); ++i)
    {
        output.data[i] = std::max(0.0f, input.data[i]);
    }
}

void ReLU::backward(const Tensor &dout, Tensor &din)
{
    din = Tensor(dout.c, dout.h, dout.w);
    for (size_t i = 0; i < dout.data.size(); ++i)
    {
        // Đạo hàm ReLU: 1 nếu x > 0, ngược lại 0
        din.data[i] = (input_cache.data[i] > 0) ? dout.data[i] : 0.0f;
    }
}

// ================== MaxPool2D ==================

MaxPool2D::MaxPool2D(int size, int s) : pool_size(size), stride(s) {}

void MaxPool2D::forward(const Tensor &input, Tensor &output)
{
    input_shape_cache = input;
    int out_h = (input.h - pool_size) / stride + 1;
    int out_w = (input.w - pool_size) / stride + 1;

    output = Tensor(input.c, out_h, out_w);
    index_mask = Tensor(input.c, out_h, out_w); // Lưu vị trí max

    for (int c = 0; c < input.c; ++c)
    {
        for (int h = 0; h < out_h; ++h)
        {
            for (int w = 0; w < out_w; ++w)
            {

                float max_val = -1e9;
                int max_idx = -1;

                // Quét cửa sổ pool
                for (int ph = 0; ph < pool_size; ++ph)
                {
                    for (int pw = 0; pw < pool_size; ++pw)
                    {
                        int in_row = h * stride + ph;
                        int in_col = w * stride + pw;

                        float val = input(c, in_row, in_col);
                        if (val > max_val)
                        {
                            max_val = val;
                            max_idx = in_row * input.w + in_col;
                        }
                    }
                }
                output(c, h, w) = max_val;
                index_mask(c, h, w) = (float)max_idx; 
            }
        }
    }
}

void MaxPool2D::backward(const Tensor &dout, Tensor &din)
{
    din = Tensor(input_shape_cache.c, input_shape_cache.h, input_shape_cache.w); // Init with 0

    for (int c = 0; c < dout.c; ++c)
    {
        for (int h = 0; h < dout.h; ++h)
        {
            for (int w = 0; w < dout.w; ++w)
            {
                int max_idx = (int)index_mask(c, h, w);

                int offset = c * din.h * din.w + max_idx;
                din.data[offset] += dout(c, h, w);
            }
        }
    }
}

// ================== UpSample2D ==================

UpSample2D::UpSample2D(int scale) : scale_factor(scale) {}

void UpSample2D::forward(const Tensor &input, Tensor &output)
{
    int out_h = input.h * scale_factor;
    int out_w = input.w * scale_factor;
    output = Tensor(input.c, out_h, out_w);

    // Nearest Neighbor
    for (int c = 0; c < output.c; ++c)
    {
        for (int h = 0; h < out_h; ++h)
        {
            for (int w = 0; w < out_w; ++w)
            {
                int in_h = h / scale_factor;
                int in_w = w / scale_factor;
                output(c, h, w) = input(c, in_h, in_w);
            }
        }
    }
}

void UpSample2D::backward(const Tensor &dout, Tensor &din)
{
    int in_h = dout.h / scale_factor;
    int in_w = dout.w / scale_factor;
    din = Tensor(dout.c, in_h, in_w);

    for (int c = 0; c < dout.c; ++c)
    {
        for (int h = 0; h < dout.h; ++h)
        {
            for (int w = 0; w < dout.w; ++w)
            {
                int in_r = h / scale_factor;
                int in_c = w / scale_factor;
                din(c, in_r, in_c) += dout(c, h, w);
            }
        }
    }
}