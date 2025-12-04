#include "cifar_loader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

CIFAR10Loader::CIFAR10Loader(const std::string& folder)
    : folder_path(folder) {}

void CIFAR10Loader::load_batch(const std::string& filename,
                               std::vector<float>& images,
                               std::vector<int>& labels,
                               int expected_images) {
    std::string path = folder_path + "/" + filename;
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    const int record_size = 1 + CIFAR_IMAGE_SIZE;
    std::vector<unsigned char> buffer(record_size);

    for (int i = 0; i < expected_images; ++i) {
        file.read((char*)buffer.data(), record_size);

        int label = buffer[0];
        labels.push_back(label);

        for (int j = 0; j < CIFAR_IMAGE_SIZE; ++j) {
            images.push_back(buffer[1 + j] / 255.0f);
        }
    }
}

void CIFAR10Loader::load_dataset(CIFAR10Dataset& dataset) {
    for (int i = 1; i <= 5; i++) {
        load_batch("data_batch_" + std::to_string(i) + ".bin",
                   dataset.train_images, dataset.train_labels, 10000);
    }
    load_batch("test_batch.bin", dataset.test_images, dataset.test_labels, 10000);

    std::cout << "Loaded CIFAR-10 dataset." << std::endl;
}
