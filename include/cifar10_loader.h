#ifndef CIFAR10_LOADER_H
#define CIFAR10_LOADER_H

#include <vector>
#include <string>

// CIFAR-10 Data Loader
// Format: Mỗi file binary chứa 10,000 images
// Mỗi image: 1 byte label + 3072 bytes (32x32x3 RGB, uint8)
// Total per file: 10,000 * (1 + 3072) = 30,730,000 bytes

class CIFAR10Loader {
public:
    // Constructor: nhận đường dẫn đến folder chứa CIFAR-10 binary files
    CIFAR10Loader(const std::string& data_dir);

    // Load tất cả training images (5 batches = 50,000 images)
    // Output: images (50,000 x 3 x 32 x 32), labels (50,000)
    // Images đã normalize [0,255] → [0,1] (float)
    void load_train_data(std::vector<float>& images, std::vector<int>& labels);

    // Load test images (1 batch = 10,000 images)
    void load_test_data(std::vector<float>& images, std::vector<int>& labels);

    // Get batch: trả về batch_size images bắt đầu từ idx
    // images: output buffer (batch_size x 3 x 32 x 32)
    // labels: output labels (batch_size) - có thể nullptr nếu không cần
    void get_batch(const std::vector<float>& all_images,
                   const std::vector<int>& all_labels,
                   std::vector<float>& batch_images,
                   std::vector<int>& batch_labels,
                   int batch_size,
                   int start_idx);

    // Shuffle data (dùng cho training)
    void shuffle_data(std::vector<float>& images, std::vector<int>& labels);

    // Constants
    static const int IMAGE_SIZE = 32 * 32 * 3;  // 3072
    static const int NUM_TRAIN = 50000;
    static const int NUM_TEST = 10000;
    static const int NUM_CLASSES = 10;

private:
    std::string data_dir_;

    // Load một file binary CIFAR-10
    // filename: "data_batch_1.bin" hoặc "test_batch.bin"
    // num_images: số images trong file (thường 10,000)
    void load_batch_file(const std::string& filename,
                        std::vector<float>& images,
                        std::vector<int>& labels,
                        int num_images);
};

#endif // CIFAR10_LOADER_H

