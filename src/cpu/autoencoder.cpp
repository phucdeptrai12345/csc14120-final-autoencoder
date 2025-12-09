#include "../../include/autoencoder.h"
#include <iostream>
#include <fstream>

Autoencoder::Autoencoder()
{
    // ENCODER ARCHITECTURE [cite: 174-191]
    // Input: 32x32x3
    enc_conv1 = new Conv2D(3, 256, 3, 1, 1);
    enc_relu1 = new ReLU();
    enc_pool1 = new MaxPool2D(2, 2); // -> 16x16x256

    enc_conv2 = new Conv2D(256, 128, 3, 1, 1);
    enc_relu2 = new ReLU();
    enc_pool2 = new MaxPool2D(2, 2); // -> 8x8x128 (Latent)

    // DECODER ARCHITECTURE [cite: 192-207]
    dec_conv1 = new Conv2D(128, 128, 3, 1, 1);
    dec_relu1 = new ReLU();
    dec_up1 = new UpSample2D(2); // -> 16x16x128

    dec_conv2 = new Conv2D(128, 256, 3, 1, 1);
    dec_relu2 = new ReLU();
    dec_up2 = new UpSample2D(2); // -> 32x32x256

    // Lớp cuối cùng: 3 filters để ra ảnh RGB, không activation (linear) hoặc Sigmoid
    // Đề bài không yêu cầu Sigmoid rõ ràng, nhưng input [0,1] nên output cũng nên thế.
    // Ở đây ta dùng Conv linear, MSE sẽ tự ép nó về đúng khoảng.
    dec_conv3 = new Conv2D(256, 3, 3, 1, 1);
}

Autoencoder::~Autoencoder()
{
    delete enc_conv1;
    delete enc_relu1;
    delete enc_pool1;
    delete enc_conv2;
    delete enc_relu2;
    delete enc_pool2;
    delete dec_conv1;
    delete dec_relu1;
    delete dec_up1;
    delete dec_conv2;
    delete dec_relu2;
    delete dec_up2;
    delete dec_conv3;
}

void Autoencoder::forward(const Tensor &input, Tensor &output)
{
    // Các biến trung gian (cần thiết nếu muốn quản lý bộ nhớ tốt hơn,
    // ở đây khai báo cục bộ, vector sẽ tự giải phóng, nhưng tốn chi phí cấp phát lại)
    // Để tối ưu phase 1, ta cứ để nó tự động.

    Tensor t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;

    // Encoder
    enc_conv1->forward(input, t1);
    enc_relu1->forward(t1, t2);
    enc_pool1->forward(t2, t3); // 16x16

    enc_conv2->forward(t3, t4);
    enc_relu2->forward(t4, t5);
    enc_pool2->forward(t5, t6); // 8x8 Latent

    // Decoder
    dec_conv1->forward(t6, t7);
    dec_relu1->forward(t7, t8);
    dec_up1->forward(t8, t9); // 16x16

    dec_conv2->forward(t9, t1); // Tái sử dụng biến t1
    dec_relu2->forward(t1, t2);
    dec_up2->forward(t2, t3); // 32x32

    dec_conv3->forward(t3, output); // Output final
}

void Autoencoder::backward(const Tensor &input, const Tensor &output, float lr)
{
    // 1. Tính đạo hàm của Loss (MSE) w.r.t Output
    // Loss = (output - input)^2
    // dLoss/dOutput = 2 * (output - input) / N (bỏ qua 1/N hằng số cũng được hoặc gộp vào lr)
    Tensor d_out(output.c, output.h, output.w);
    for (size_t i = 0; i < d_out.data.size(); ++i)
    {
        d_out.data[i] = 2.0f * (output.data[i] - input.data[i]);
    }

    Tensor d1, d2, d3, d4, d5, d6, d7, d8, d9;

    // Backprop ngược lại với quy trình forward

    // Decoder Backward
    dec_conv3->backward(d_out, d1, lr);

    dec_up2->backward(d1, d2);
    dec_relu2->backward(d2, d3);
    dec_conv2->backward(d3, d4, lr);

    dec_up1->backward(d4, d5);
    dec_relu1->backward(d5, d6);
    dec_conv1->backward(d6, d7, lr);

    // Encoder Backward
    enc_pool2->backward(d7, d8);
    enc_relu2->backward(d8, d9);
    enc_conv2->backward(d9, d1, lr); // reuse d1

    enc_pool1->backward(d1, d2);
    enc_relu1->backward(d2, d3);
    enc_conv1->backward(d3, d4, lr); // d4 là gradient tại input ảnh gốc (không dùng)
}

float Autoencoder::compute_loss(const Tensor &original, const Tensor &reconstructed)
{
    float sum_sq = 0.0f;
    for (size_t i = 0; i < original.data.size(); ++i)
    {
        float diff = original.data[i] - reconstructed.data[i];
        sum_sq += diff * diff;
    }
    return sum_sq / original.data.size(); // Mean Squared Error
}

void Autoencoder::save_weights(const std::string &filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    // Hàm phụ để ghi vector
    auto write_vec = [&](const std::vector<float> &v)
    {
        size_t size = v.size();
        file.write((char *)&size, sizeof(size));
        file.write((char *)v.data(), size * sizeof(float));
    };

    write_vec(enc_conv1->weights);
    write_vec(enc_conv1->biases);
    write_vec(enc_conv2->weights);
    write_vec(enc_conv2->biases);
    write_vec(dec_conv1->weights);
    write_vec(dec_conv1->biases);
    write_vec(dec_conv2->weights);
    write_vec(dec_conv2->biases);
    write_vec(dec_conv3->weights);
    write_vec(dec_conv3->biases);

    file.close();
    std::cout << "Saved weights to " << filepath << std::endl;
}

void Autoencoder::load_weights(const std::string &filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open model file " << filepath << std::endl;
        return;
    }

    // Hàm phụ để đọc vector (Ngược lại với write_vec)
    auto read_vec = [&](std::vector<float> &v)
    {
        size_t size;
        file.read((char *)&size, sizeof(size));
        // Resize vector để chứa đủ dữ liệu
        v.resize(size);
        file.read((char *)v.data(), size * sizeof(float));
    };

    read_vec(enc_conv1->weights);
    read_vec(enc_conv1->biases);
    read_vec(enc_conv2->weights);
    read_vec(enc_conv2->biases);

    read_vec(dec_conv1->weights);
    read_vec(dec_conv1->biases);
    read_vec(dec_conv2->weights);
    read_vec(dec_conv2->biases);
    read_vec(dec_conv3->weights);
    read_vec(dec_conv3->biases);

    file.close();
    std::cout << "Loaded weights from " << filepath << std::endl;
}

void Autoencoder::extract_features(const Tensor &input, std::vector<float> &features)
{
    // Biến trung gian cục bộ để chạy Encoder
    Tensor t1, t2, t3, t4, t5, latent;

    // Chỉ chạy luồng Encoder (Forward pass nửa đầu)
    enc_conv1->forward(input, t1);
    enc_relu1->forward(t1, t2);
    enc_pool1->forward(t2, t3);

    enc_conv2->forward(t3, t4);
    enc_relu2->forward(t4, t5);
    enc_pool2->forward(t5, latent); // Kết quả là Tensor (128, 8, 8)

    // Flatten (Duỗi phẳng) Tensor thành Vector cho SVM
    // Latent size = 128 * 8 * 8 = 8192
    features = latent.data;
}