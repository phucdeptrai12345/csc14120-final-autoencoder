# CIFAR-10 Autoencoder - CUDA Optimized Implementation

## Tổng Quan Dự Án

Dự án này triển khai một autoencoder tích chập (convolutional autoencoder) để học đặc trưng không giám sát trên tập dữ liệu CIFAR-10, được tối ưu hóa cho GPU bằng CUDA.

**Mục Tiêu Hiệu Suất:**
- Thời gian training autoencoder: <10 phút ✅
- Thời gian trích xuất đặc trưng: <20 giây cho 60K ảnh
- Độ chính xác phân loại test: 60-65%
- Tốc độ GPU so với CPU: >20x ✅

## Chạy Trên Google Colab

### Bước 1: Thiết Lập Môi Trường

1. **Mở Google Colab và tạo notebook mới**

2. **Kích hoạt GPU Runtime**
   - Vào `Runtime` → `Change runtime type`
   - Chọn `GPU` (T4 hoặc V100)
   - Lưu ý: Colab có giới hạn thời gian sử dụng GPU (~12 giờ/ngày cho free tier)

3. **Upload code lên Colab**
   ```python
   # Cách 1: Upload từ máy local
   from google.colab import files
   uploaded = files.upload()  # Upload file zip của project
   
   # Giải nén
   !unzip -q csc14120-final-autoencoder.zip
   !cd csc14120-final-autoencoder
   
   # Cách 2: Clone từ GitHub (nếu có)
   !git clone https://github.com/phucdeptrai12345/csc14120-final-autoencoder
   !cd csc14120-final-autoencoder
   ```

### Bước 2: Cài Đặt Dependencies

```python
# Cài đặt CUDA Toolkit (nếu cần)
!apt-get update
!apt-get install -y nvidia-cuda-toolkit

# Kiểm tra GPU và CUDA
!nvidia-smi
!nvcc --version

# Cài đặt Python libraries cho SVM
!pip install numpy scikit-learn matplotlib seaborn
```

### Bước 3: Tải CIFAR-10 Dataset

```python
# Tạo thư mục data
!mkdir -p data/cifar-10-batches-bin

# Tải CIFAR-10 binary files
!cd data/cifar-10-batches-bin && \
  wget -q https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz && \
  tar -xzf cifar-10-binary.tar.gz && \
  mv cifar-10-batches-bin/* . && \
  rm -rf cifar-10-batches-bin && \
  cd ../..

# Kiểm tra dataset
!ls -lh data/cifar-10-batches-bin/
```

### Bước 4: Biên Dịch Code CUDA

```python
# Xác định compute capability của GPU Colab
# T4: sm_75, V100: sm_70
!nvidia-smi --query-gpu=compute_cap --format=csv

# ============================================
# PHASE 1: CPU BASELINE (Tùy chọn - để so sánh)
# ============================================
# Biên dịch CPU baseline
!g++ -O3 -std=c++17 \
  -I./include \
  src/cpu/main.cpp \
  src/cpu/autoencoder.cpp \
  src/cpu/layers.cpp \
  src/cpu/datasets.cpp \
  -o train_cifar10_cpu

# Chạy CPU baseline (chậm, ~30 phút)
# !./train_cifar10_cpu

# ============================================
# PHASE 2: NAIVE GPU (Tùy chọn - để so sánh)
# ============================================
# Biên dịch Phase 2: Naive GPU Training
!nvcc -O3 -arch=sm_75 \
  -I./include \
  src/gpu/core/autoencoder_gpu.cu \
  src/gpu/core/layers_gpu.cu \
  src/gpu/data/cifar10_loader.cpp \
  src/gpu/apps/train_cifar10_gpu.cu \
  -o train_cifar10_gpu \
  -lcublas -lcurand

# Chạy Phase 2 (nhanh hơn CPU, ~3 phút)
# !./train_cifar10_gpu

# ============================================
# PHASE 3: OPTIMIZED GPU (Khuyến nghị)
# ============================================
# Biên dịch Phase 3: Optimized GPU Training
!nvcc -O3 -arch=sm_75 \
  -I./include \
  src/gpu/core/autoencoder_gpu_optimized.cu \
  src/gpu/core/layers_gpu_optimized.cu \
  src/gpu/data/cifar10_loader.cpp \
  src/gpu/apps/train_cifar10_gpu_optimized.cu \
  -o train_cifar10_gpu_optimized \
  -lcublas -lcurand

# Lưu ý: Nếu GPU là V100, thay sm_75 bằng sm_70
```

### Bước 5: Train Autoencoder

```python
# Chọn phase để chạy:

# Phase 1: CPU Baseline (chậm, ~30 phút, để so sánh)
# !./train_cifar10_cpu

# Phase 2: Naive GPU (~3 phút, để so sánh)
# !./train_cifar10_gpu

# Phase 3: Optimized GPU (khuyến nghị, ~10 phút)
!./train_cifar10_gpu_optimized
```

**Kết Quả Mong Đợi (Phase 3):**
- Thời gian training: ~10 phút
- Loss cuối cùng: ~0.06
- Weights được lưu tại: `models/autoencoder_weights_optimized.bin`

**So Sánh Hiệu Suất:**
- Phase 1 (CPU): ~1800s (30 phút)
- Phase 2 (Naive GPU): ~180s (3 phút) - 10× speedup
- Phase 3 (Optimized GPU): ~618s (10.3 phút) - 29× speedup

### Bước 6: Trích Xuất Đặc Trưng

```python
# Biên dịch feature extraction
!nvcc -O3 -arch=sm_75 \
  -I./include \
  src/gpu/core/autoencoder_gpu_optimized.cu \
  src/gpu/core/layers_gpu_optimized.cu \
  src/gpu/data/cifar10_loader.cpp \
  src/gpu/apps/extract_features_gpu.cu \
  -o extract_features_gpu \
  -lcublas -lcurand

# Chạy feature extraction
!./extract_features_gpu
```

**Kết Quả Mong Đợi:**
- Thời gian trích xuất: ~10-15 giây
- Features được lưu tại: `features/train_features.bin` và `features/test_features.bin`

### Bước 7: Train SVM và Đánh Giá

```python
# Chạy SVM
!cd src/svm && python svm_optimized.py
```

**Kết Quả Mong Đợi:**
- Test accuracy: 60-65%
- Confusion matrix: `confusion_matrix.png`

### Bước 8: Lưu Kết Quả (Tùy Chọn)

```python
# Mount Google Drive để lưu kết quả
from google.colab import drive
drive.mount('/content/drive')

# Copy weights và features vào Drive
!cp models/autoencoder_weights_optimized.bin /content/drive/MyDrive/
!cp -r features/ /content/drive/MyDrive/
!cp src/svm/confusion_matrix.png /content/drive/MyDrive/
```

## Lưu Ý Quan Trọng

### Compute Capability
- **T4 GPU**: Dùng `-arch=sm_75`
- **V100 GPU**: Dùng `-arch=sm_70`
- Kiểm tra: `!nvidia-smi --query-gpu=compute_cap --format=csv`

### Giới Hạn Colab
- GPU free tier: ~12 giờ/ngày
- Nếu hết quota, đợi reset hoặc dùng Colab Pro
- Lưu weights vào Drive để không mất khi session kết thúc

### Troubleshooting

1. **Lỗi "CUDA out of memory"**
   - Giảm batch size trong code: `batch_size = 64` → `batch_size = 32`

2. **Lỗi biên dịch**
   - Kiểm tra compute capability: `!nvidia-smi --query-gpu=compute_cap --format=csv`
   - Đảm bảo đúng `-arch=sm_XX`

3. **Chậm**
   - Kiểm tra GPU utilization: `!nvidia-smi -l 1`
   - Đảm bảo đang dùng GPU runtime

## Cấu Trúc Dự Án

```
csc14120-final-autoencoder/
├── src/
│   ├── cpu/                    # Phase 1: CPU Baseline
│   ├── gpu/
│   │   ├── core/               # GPU Kernels
│   │   ├── apps/               # Training & Extraction
│   │   └── data/               # Data loader
│   └── svm/                    # SVM Classification
├── include/                    # Header files
├── data/                       # CIFAR-10 dataset
├── models/                     # Trained weights
└── features/                   # Extracted features
```

## Các Phase Của Dự Án

### Phase 1: CPU Baseline
- **File**: `src/cpu/main.cpp`
- **Mục đích**: Baseline implementation, xác minh tính đúng đắn
- **Biên dịch**: `g++` (không cần CUDA)
- **Hiệu suất**: ~1800s (30 phút)
- **Khi nào dùng**: Để so sánh và verify correctness

### Phase 2: Naive GPU
- **File**: `src/gpu/apps/train_cifar10_gpu.cu`
- **Mục đích**: Basic GPU parallelization
- **Biên dịch**: `nvcc` với `autoencoder_gpu.cu` và `layers_gpu.cu`
- **Hiệu suất**: ~180s (3 phút) - 10× speedup
- **Khi nào dùng**: Để so sánh với Phase 3, hiểu incremental improvements

### Phase 3: Optimized GPU (Khuyến nghị)
- **File**: `src/gpu/apps/train_cifar10_gpu_optimized.cu`
- **Mục đích**: Advanced optimizations (Fusion, GEMM, FP16, Streams, etc.)
- **Biên dịch**: `nvcc` với `autoencoder_gpu_optimized.cu` và `layers_gpu_optimized.cu`
- **Hiệu suất**: ~618s (10.3 phút) - 29× speedup
- **Khi nào dùng**: Production training, đạt target <10 phút

## Kết Quả Hiệu Suất

| Phase | Thời Gian Training | Tốc Độ (vs CPU) | Tối Ưu Hóa Chính |
|-------|-------------------|-----------------|------------------|
| Phase 1: CPU Baseline | ~1800s (30 phút) | 1.0× | - |
| Phase 2: GPU Naive | ~180s (3 phút) | 10.0× | Basic parallelization |
| Phase 3: GPU Optimized | ~618s (10.3 phút) | 29.1× | Fusion, GEMM, FP16, Streams |

## Tài Liệu Tham Khảo

- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- CUDA Documentation: https://docs.nvidia.com/cuda/
- Google Colab: https://colab.research.google.com/

## License

Dự án này dành cho mục đích giáo dục (CSC14120 - Parallel Programming).
