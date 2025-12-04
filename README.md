# CSC14120 Final Project - Autoencoder + CUDA + SVM

## Project structure (clean restart)

- src/cpu      : CPU implementation (layers, autoencoder, training) - **Quan**
- src/gpu      : GPU implementation (CUDA kernels, optimized autoencoder) - **Phuc**
- include      : header files shared between CPU/GPU
- data         : CIFAR-10 dataset (local only, NOT pushed to GitHub)
- models       : saved weights/features (local only)

## Task split

### Phuc (GPU & performance)
- Design GPU layers in `src/gpu`:
  - Conv2D, ReLU, MaxPool, Upsample (forward + backward)
- Implement GPU autoencoder training:
  - naive version (Phase 2)
  - optimized version (Phase 3: shared memory, kernel fusion, etc.)
- Measure and report speedup:
  - CPU vs GPU basic vs GPU optimized

### Quan (CPU, SVM & report)
- Implement CPU baseline in `src/cpu`:
  - Conv2D, ReLU, MaxPool, Upsample
  - AutoencoderCPU class (forward, backward, update)
  - Training loop, logging loss, reconstruction samples
- Feature extraction + SVM:
  - Extract encoder features for CIFAR-10
  - Train SVM classifier and evaluate accuracy
- Colab notebook report:
  - Figures (loss curves, speedup plots, reconstructions, confusion matrix)
  - Written analysis and conclusion

## Data (CIFAR-10)

We will **not** commit the dataset to GitHub to avoid size limits.
Each environment (local or Colab) will download CIFAR-10 at runtime into `data/`.

