#include <iostream>
#include <vector>
#include <string>
#include <chrono>     // Thư viện đo thời gian
#include <iomanip>    // Thư viện định dạng in ấn
#include <filesystem> // C++17

#include "../../include/dataset.h"
#include "../../include/autoencoder.h"

// --- 1. CẤU HÌNH ĐỂ TEST NHANH ---
const int BATCH_SIZE = 32;
// Giảm số epoch xuống 5 để test nhanh (thay vì 20)
const int EPOCHS = 5;
const float LEARNING_RATE = 0.00001f;

// [QUAN TRỌNG] Giới hạn số lượng ảnh để train (thay vì 50000)
// 100 ảnh là đủ để kiểm tra code chạy đúng hay sai trong 1-2 phút
const int DEBUG_LIMIT = 100;

int main()
{
    // Tạo thư mục models nếu chưa có
    if (!std::filesystem::exists("models"))
    {
        std::filesystem::create_directory("models");
    }

    // --- 2. DATA LOADING ---
    std::cout << "Phase 1: CPU Baseline (DEBUG MODE - FAST RUN)" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "[1/4] Loading CIFAR-10 Dataset..." << std::endl;

    CIFAR10Dataset dataset;
    dataset.load_train("data");

    if (dataset.train_images.empty())
    {
        std::cerr << "Error: No training data loaded!" << std::endl;
        return 1;
    }

    // [CẮT DỮ LIỆU ĐỂ CHẠY NHANH]
    if (dataset.train_images.size() > DEBUG_LIMIT)
    {
        std::cout << "WARNING: Resizing dataset to " << DEBUG_LIMIT
                  << " images for fast testing on CPU." << std::endl;

        // Cắt bớt vector dữ liệu
        dataset.train_images.resize(DEBUG_LIMIT);
        dataset.train_labels.resize(DEBUG_LIMIT);

        // Cập nhật lại vector chỉ số (indices) cho bộ shuffle
        // Bắt buộc phải clear để shuffle_data() tự tạo lại đúng kích thước mới
        dataset.indices.clear();
    }

    // --- 3. MODEL INITIALIZATION ---
    std::cout << "[2/4] Initializing Autoencoder..." << std::endl;
    Autoencoder model;

    std::cout << "      Hyperparameters: Batch=" << BATCH_SIZE
              << ", Epochs=" << EPOCHS
              << ", LR=" << LEARNING_RATE
              << ", Data Limit=" << DEBUG_LIMIT << std::endl;

    // --- 4. TRAINING LOOP ---
    std::cout << "[3/4] Starting Training Loop..." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "| Epoch |   Avg Loss   |  Time (s)  |  Images/Sec |" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Xáo trộn dữ liệu (trên tập đã cắt nhỏ)
        dataset.shuffle_data();

        float total_epoch_loss = 0.0f;
        int total_images = 0;

        std::vector<Tensor> batch_images;

        // Vòng lặp Batch
        while (dataset.get_next_batch(BATCH_SIZE, batch_images))
        {
            float batch_loss = 0.0f;

            for (const auto &img : batch_images)
            {
                Tensor output;
                // Forward
                model.forward(img, output);

                // Loss
                float loss = model.compute_loss(img, output);
                batch_loss += loss;

                // Backward & Update
                model.backward(img, output, LEARNING_RATE);
            }

            total_epoch_loss += batch_loss;
            total_images += batch_images.size();

            // In dấu chấm để biết chương trình đang chạy (không bị treo)
            std::cout << "." << std::flush;
        }
        std::cout << "\r"; // Xóa dòng dấu chấm

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        float avg_loss = total_epoch_loss / total_images;
        float img_per_sec = total_images / elapsed.count();

        std::cout << "|  " << std::setw(3) << (epoch + 1) << "  | "
                  << std::fixed << std::setprecision(6) << avg_loss << " | "
                  << std::setw(9) << std::setprecision(2) << elapsed.count() << "s | "
                  << std::setw(10) << img_per_sec << " |" << std::endl;
    }
    std::cout << "------------------------------------------------------------" << std::endl;

    // --- 5. SAVE MODEL ---
    std::cout << "[4/4] Saving trained weights..." << std::endl;
    std::string model_path = "models/cifar10_ae_cpu_debug.bin";
    model.save_weights(model_path);

    std::cout << "      Model saved to: " << model_path << std::endl;
    std::cout << "Phase 1 Test Completed!" << std::endl;

    return 0;
}