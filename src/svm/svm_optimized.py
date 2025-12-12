import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # <--- THÊM CÁI NÀY ĐỂ TĂNG TỐC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os

# ==========================================
# CẤU HÌNH ĐỂ CHẠY CỰC NHANH (TURBO MODE)
# ==========================================
LIMIT_TRAIN = 5000   # Lấy 5000 mẫu để train
LIMIT_TEST = 1000    # Lấy 1000 mẫu để test
N_COMPONENTS = 128   # Nén từ 8192 chiều xuống 128 chiều (Quan trọng nhất)

def load_features_optimized(filepath, limit=None):
    """
    Load dữ liệu thông minh: Nhảy cóc qua phần thừa, không load hết vào RAM.
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found!")
        return None, None

    print(f"Loading {filepath} (Lấy {limit} dòng)...", end=" ", flush=True)
    
    try:
        with open(filepath, 'rb') as f:
            # 1. Đọc Header
            header = np.fromfile(f, dtype=np.int32, count=2)
            total_samples = header[0]
            feature_dim = header[1]
            
            num_to_read = total_samples
            if limit is not None and limit < total_samples:
                num_to_read = limit
            
            # 2. Đọc Features (Chỉ đọc số lượng cần thiết)
            features = np.fromfile(f, dtype=np.float32, count=num_to_read * feature_dim)
            features = features.reshape((num_to_read, feature_dim))
            
            # 3. Nhảy đến vùng Labels
            # Header (8 bytes) + Toàn bộ vùng features gốc (samples * dim * 4 bytes)
            labels_start_offset = 8 + (total_samples * feature_dim * 4)
            f.seek(labels_start_offset)
            
            # 4. Đọc Labels tương ứng
            labels = np.fromfile(f, dtype=np.int32, count=num_to_read)
            
        print(f"Done. Shape: {features.shape}")
        return features, labels
        
    except Exception as e:
        print(f"\nLỗi đọc file: {e}")
        return None, None

def main():
    total_start = time.time()
    
    # --- BƯỚC 1: LOAD DỮ LIỆU ---
    # Sửa lại đường dẫn '../models/...' cho đúng với thư mục máy bạn
    X_train, y_train = load_features_optimized('../../models/train_features.bin', limit=LIMIT_TRAIN)
    X_test, y_test = load_features_optimized('../../models/test_features.bin', limit=LIMIT_TEST)

    if X_train is None or X_test is None:
        return

    # --- BƯỚC 2: GIẢM CHIỀU DỮ LIỆU (PCA) - CHÌA KHÓA TĂNG TỐC ---
    print(f"\n[OPTIMIZATION] Đang nén dữ liệu từ {X_train.shape[1]} xuống {N_COMPONENTS} chiều...")
    pca_start = time.time()
    
    # PCA giúp vứt bỏ các thông tin nhiễu, giữ lại đặc trưng quan trọng nhất
    pca = PCA(n_components=N_COMPONENTS, whiten=True) 
    
    # Fit trên tập train và transform cả 2 tập
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print(f"PCA xong trong {time.time() - pca_start:.2f}s. New shape: {X_train.shape}")

    # --- BƯỚC 3: TRAIN SVM (CẤU HÌNH NHANH) ---
    print("\nStarting SVM Training...")
    svm_start = time.time()

    # Tinh chỉnh tham số để chạy nhanh:
    # - max_iter=200: Chỉ lặp 200 lần (đủ để xem kết quả sơ bộ)
    # - tol=1e-2: Chấp nhận sai số lớn hơn một chút để dừng sớm
    clf = LinearSVC(C=0.1, dual=False, max_iter=200, tol=1e-2, verbose=1)
    
    clf.fit(X_train, y_train)
    
    print(f"Training xong trong {time.time() - svm_start:.2f}s.")

    # --- BƯỚC 4: KẾT QUẢ ---
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"========================================")
    print(f"ACCURACY (Fast Check): {acc * 100:.2f}%")
    print(f"========================================")
    
    # Lưu ảnh kết quả
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'SVM Result (PCA={N_COMPONENTS}, Acc={acc*100:.1f}%)')
    plt.savefig('svm_result_fast.png')
    print("Đã lưu biểu đồ vào 'svm_result_fast.png'")
    
    print(f"\nTỔNG THỜI GIAN CHẠY: {time.time() - total_start:.2f} giây")

if __name__ == "__main__":
    main()