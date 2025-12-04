#include <iostream>
#include "cifar_loader.h"

int main() {
    CIFAR10Dataset dataset;
    CIFAR10Loader loader("data/cifar-10-batches-bin");

    loader.load_dataset(dataset);

    std::cout << "Train images: " << dataset.train_images.size() << std::endl;
    std::cout << "Train labels: " << dataset.train_labels.size() << std::endl;
    std::cout << "Test images: " << dataset.test_images.size() << std::endl;
    std::cout << "Test labels: " << dataset.test_labels.size() << std::endl;
}
