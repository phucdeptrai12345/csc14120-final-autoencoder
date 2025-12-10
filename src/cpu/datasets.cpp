#include "../../include/dataset.h"
#include <fstream>
#include <iostream>
#include <memory>    
#include <algorithm> 
#include <random> 

void CIFAR10Dataset::load_train(const std::string &data_folder)
{
    train_images.clear();
    train_labels.clear();

    // Dự phòng bộ nhớ (50.000 ảnh)
    train_images.reserve(50000);
    train_labels.reserve(50000);

    for (int i = 1; i <= 5; ++i)
    {
        std::string filename = data_folder + "/data_batch_" + std::to_string(i) + ".bin";
        std::cout << "Loading " << filename << "..." << std::endl;
        read_binary_file(filename, train_images, train_labels);
    }
    std::cout << "Loaded " << train_images.size() << " training images." << std::endl;
}

void CIFAR10Dataset::load_test(const std::string &data_folder)
{
    test_images.clear();
    test_labels.clear();

    test_images.reserve(10000);
    test_labels.reserve(10000);

    std::string filename = data_folder + "/test_batch.bin";
    std::cout << "Loading " << filename << "..." << std::endl;
    read_binary_file(filename, test_images, test_labels);
    std::cout << "Loaded " << test_images.size() << " test images." << std::endl;
}

void CIFAR10Dataset::read_binary_file(const std::string &filename, std::vector<Tensor> &images, std::vector<unsigned char> &labels)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // 1. Lấy kích thước file
    auto file_size = file.tellg();

    // 2. Tạo buffer chứa toàn bộ file
    std::unique_ptr<char[]> buffer(new char[file_size]);

    // 3. Đọc một lần duy nhất (Tối ưu I/O)
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), file_size);
    file.close();

    // 4. Parse dữ liệu
    const int num_images = 10000;
    const int entry_size = 3073; // 1 byte label + 3072 bytes pixels

    for (int i = 0; i < num_images; ++i)
    {
        // Start image byte
        int start_idx = i * entry_size;

        // Byte label
        unsigned char label = (unsigned char)buffer[start_idx];
        labels.push_back(label);

        Tensor img(3, 32, 32);

        // First Byte 
        char *img_buffer_ptr = &buffer[start_idx + 1];

        // Check size
        if (img.data.size() != 3072)
        {
            img.data.resize(3072);
        }

        for (int k = 0; k < 3072; ++k)
        {
            unsigned char pixel_byte = (unsigned char)img_buffer_ptr[k];

            img.data[k] = static_cast<float>(pixel_byte) / 255.0f;
        }

        images.push_back(img);
    }
}

void CIFAR10Dataset::shuffle_data()
{
    if (indices.size() != train_images.size())
    {
        indices.resize(train_images.size());
        for (size_t i = 0; i < indices.size(); ++i)
            indices[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    current_batch_idx = 0;
}

bool CIFAR10Dataset::get_next_batch(int batch_size, std::vector<Tensor> &batch_images)
{
    batch_images.clear();

    if (current_batch_idx >= indices.size())
        return false;

    int count = 0;
    while (count < batch_size && current_batch_idx < indices.size())
    {
        int original_idx = indices[current_batch_idx];
        batch_images.push_back(train_images[original_idx]);
        current_batch_idx++;
        count++;
    }

    return true;
}