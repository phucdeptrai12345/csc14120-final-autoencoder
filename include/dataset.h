#ifndef DATASET_H
#define DATASET_H

#include "utils.h"
#include <string>
#include <vector>
#include <algorithm> 
#include <random> 

class CIFAR10Dataset
{
public:
    std::vector<Tensor> train_images;
    std::vector<unsigned char> train_labels;

    std::vector<Tensor> test_images;
    std::vector<unsigned char> test_labels;

    // Support Batch
    std::vector<int> indices;  // Shuff Array
    int current_batch_idx = 0;

    CIFAR10Dataset() {}

    void load_train(const std::string &data_folder);
    void load_test(const std::string &data_folder);

    void shuffle_data();

    bool get_next_batch(int batch_size, std::vector<Tensor> &batch_images);

private:
    void read_binary_file(const std::string &filename, std::vector<Tensor> &images, std::vector<unsigned char> &labels);
};

#endif // DATASET_H