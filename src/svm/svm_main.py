import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os

def load_binary_features(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found!")
        return None, None

    print(f"Loading {filepath}...", end=" ", flush=True)
    with open(filepath, 'rb') as f:
        # 1. Đọc Header (2 số int)
        header = np.fromfile(f, dtype=np.int32, count=2)
        num_samples = header[0]
        feature_dim = header[1]
        
        # 2. Đọc Features (Mảng float)
        # Kích thước = num_samples * feature_dim
        features = np.fromfile(f, dtype=np.float32, count=num_samples * feature_dim)
        features = features.reshape((num_samples, feature_dim))
        
        # 3. Đọc Labels (Mảng int)
        labels = np.fromfile(f, dtype=np.int32, count=num_samples)
        
    print(f"Done. Shape: {features.shape}")
    return features, labels

def main():
    # --- BƯỚC 1: LOAD DỮ LIỆU ---
    X_train, y_train = load_binary_features('../../models/train_features.bin')
    X_test, y_test = load_binary_features('../../models/test_features.bin')

    if X_train is None or X_test is None:
        return

    # --- BƯỚC 2: TRAIN SVM ---
    print("\nStarting SVM Training (Linear Kernel)...")
    start_time = time.time()

    clf = LinearSVC(C=1.0, dual=False, max_iter=1000)
    
    clf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")

    # --- BƯỚC 3: ĐÁNH GIÁ ---
    print("\nEvaluating on Test Set...")
    y_pred = clf.predict(X_test)

    # Tính Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"========================================")
    print(f"ACCURACY: {acc * 100:.2f}%")
    print(f"========================================")

    # In báo cáo chi tiết (Preciclesion, Recall, F1)
    # CIFAR-10 Class Names
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- BƯỚC 4: VẼ CONFUSION MATRIX (Điểm cộng lớn so với C++) ---
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'SVM Confusion Matrix (Acc: {acc*100:.2f}%)')
    plt.savefig('confusion_matrix.png') # Lưu ảnh ra file
    print("Confusion matrix saved to 'confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    main()