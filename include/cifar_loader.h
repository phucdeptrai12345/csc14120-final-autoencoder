#ifndef CIFAR_LOADER_H
#define CIFAR_LOADER_H

#include <string>
#include <vector>

const int CIFAR_IMAGE_SIZE = 32 * 32 * 3;
const int CIFAR_TRAIN_IMAGES = 50000;
const int CIFAR_TEST_IMAGES  = 10000;

struct CIFAR10Dataset {
    std::vector<float> train_images;
    std::vector<int>   train_labels;

    std::vector<float> test_images;
    std::vector<int>   test_labels;
};

class CIFAR10Loader {
public:
    CIFAR10Loader(const std::string& folder);
    void load_dataset(CIFAR10Dataset& dataset);

private:
    std::string folder_path;

    void load_batch(const std::string& filename,
                    std::vector<float>& images,
                    std::vector<int>& labels,
                    int expected_images);
};

#endif
