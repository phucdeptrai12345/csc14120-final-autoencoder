#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "layers.h"
#include <string>

class Autoencoder
{
public:
    // --- Các lớp của Encoder ---
    // Conv1: 3 -> 256
    Conv2D *enc_conv1;
    ReLU *enc_relu1;
    MaxPool2D *enc_pool1; // 32x32 -> 16x16

    // Conv2: 256 -> 128
    Conv2D *enc_conv2;
    ReLU *enc_relu2;
    MaxPool2D *enc_pool2; // 16x16 -> 8x8 (Latent)

    // --- Các lớp của Decoder ---
    // Conv3: 128 -> 128
    Conv2D *dec_conv1;
    ReLU *dec_relu1;
    UpSample2D *dec_up1; // 8x8 -> 16x16

    // Conv4: 128 -> 256
    Conv2D *dec_conv2;
    ReLU *dec_relu2;
    UpSample2D *dec_up2; // 16x16 -> 32x32

    // Conv5: 256 -> 3 (Output)
    Conv2D *dec_conv3;

    Autoencoder();
    ~Autoencoder();

    void forward(const Tensor &input, Tensor &output);

    void backward(const Tensor &input, const Tensor &output, float lr);

    // MSE Loss
    float compute_loss(const Tensor &original, const Tensor &reconstructed);

    void save_weights(const std::string &filepath);
    void load_weights(const std::string &filepath);


    void extract_features(const Tensor &input, std::vector<float> &features);
    
};

#endif // AUTOENCODER_H