#include "cifar10_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

CIFAR10Loader::CIFAR10Loader(const std::string& data_dir)
    : data_dir_(data_dir) {
}

void CIFAR10Loader::load_batch_file(const std::string& filename,
                                     std::vector<float>& images,
                                     std::vector<int>& labels,
                                     int num_images) {
    std::string filepath = data_dir_ + "/" + filename;
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file " << filepath << std::endl;
        std::cerr << "Please check if file exists!" << std::endl;
        exit(1);  // Dừng chương trình nếu không mở được file
    }

    // Reserve space
    size_t start_idx = images.size();
    images.resize(start_idx + num_images * IMAGE_SIZE);
    labels.resize(start_idx / IMAGE_SIZE + num_images);

    // Read từng image
    for (int i = 0; i < num_images; ++i) {
        // Đọc label (1 byte)
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        if (file.gcount() != 1) {
            std::cerr << "ERROR: Failed to read label for image " << i << " in " << filename << std::endl;
            exit(1);
        }
        labels[start_idx / IMAGE_SIZE + i] = static_cast<int>(label);

        // Đọc image (3072 bytes)
        unsigned char pixel;
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            file.read(reinterpret_cast<char*>(&pixel), 1);
            if (file.gcount() != 1) {
                std::cerr << "ERROR: Failed to read pixel " << j << " for image " << i << " in " << filename << std::endl;
                exit(1);
            }
            // Normalize [0,255] → [0,1]
            images[start_idx + i * IMAGE_SIZE + j] = static_cast<float>(pixel) / 255.0f;
        }
    }

    file.close();
    std::cout << "  ✓ Successfully loaded " << num_images << " images from " << filename << std::endl;
}

void CIFAR10Loader::load_train_data(std::vector<float>& images, std::vector<int>& labels) {
    images.clear();
    labels.clear();

    // Load 5 training batches (hoặc ít hơn nếu thiếu files)
    int batches_loaded = 0;
    for (int i = 1; i <= 5; ++i) {
        std::string filename = "data_batch_" + std::to_string(i) + ".bin";
        std::string filepath = data_dir_ + "/" + filename;
        std::ifstream test_file(filepath, std::ios::binary);
        
        if (!test_file.is_open()) {
            std::cout << "Skipping " << filename << " (file not found)" << std::endl;
            test_file.close();
            continue;  // Skip file nếu không tồn tại
        }
        test_file.close();
        
        std::cout << "Loading " << filename << "..." << std::flush;
        load_batch_file(filename, images, labels, 10000);
        batches_loaded++;
    }

    int total_loaded = images.size() / IMAGE_SIZE;
    std::cout << "\n✓ Total loaded: " << total_loaded << " training images from " 
              << batches_loaded << " batches" << std::endl;
    
    if (total_loaded != 50000) {
        std::cout << "NOTE: Using " << total_loaded << " images instead of 50000" << std::endl;
    }
}

void CIFAR10Loader::load_test_data(std::vector<float>& images, std::vector<int>& labels) {
    images.clear();
    labels.clear();

    std::cout << "Loading test_batch.bin..." << std::endl;
    load_batch_file("test_batch.bin", images, labels, 10000);
    std::cout << "Loaded " << images.size() / IMAGE_SIZE << " test images" << std::endl;
}

void CIFAR10Loader::get_batch(const std::vector<float>& all_images,
                               const std::vector<int>& all_labels,
                               std::vector<float>& batch_images,
                               std::vector<int>& batch_labels,
                               int batch_size,
                               int start_idx) {
    int total_images = all_images.size() / IMAGE_SIZE;

    if (start_idx + batch_size > total_images) {
        batch_size = total_images - start_idx;
    }

    batch_images.resize(batch_size * IMAGE_SIZE);
    batch_labels.resize(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        int img_idx = start_idx + i;
        // Copy image
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            batch_images[i * IMAGE_SIZE + j] = all_images[img_idx * IMAGE_SIZE + j];
        }
        // Copy label
        batch_labels[i] = all_labels[img_idx];
    }
}

void CIFAR10Loader::shuffle_data(std::vector<float>& images, std::vector<int>& labels) {
    int num_images = images.size() / IMAGE_SIZE;
    std::vector<int> indices(num_images);
    for (int i = 0; i < num_images; ++i) {
        indices[i] = i;
    }

    // Shuffle indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Create shuffled copies
    std::vector<float> shuffled_images(images.size());
    std::vector<int> shuffled_labels(labels.size());

    for (int i = 0; i < num_images; ++i) {
        int old_idx = indices[i];
        // Copy image
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            shuffled_images[i * IMAGE_SIZE + j] = images[old_idx * IMAGE_SIZE + j];
        }
        // Copy label
        shuffled_labels[i] = labels[old_idx];
    }

    images = shuffled_images;
    labels = shuffled_labels;
}

