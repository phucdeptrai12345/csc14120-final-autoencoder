# CSC14120 Final Project - Autoencoder + CUDA + SVM

## Current status
- Phase 1.1: CIFAR-10 data loader (CPU) is done.

## Task split

- Phúc (GPU & Optimization):
  - Implement CUDA layers (Conv2D, ReLU, MaxPool, Upsample).
  - Implement GPU autoencoder (forward + backward).
  - Optimize GPU version (shared memory, kernel fusion, etc.).
  - Measure performance: CPU vs GPU basic vs GPU optimized.

- Quân (CPU, SVM & Report):
  - Implement CPU layers and autoencoder training loop (baseline).
  - Train autoencoder on CPU and verify reconstruction quality.
  - Extract features from encoder and train SVM classifier.
  - Build Colab notebook report (plots, tables, discussion, conclusion).

## Project structure (for now)
- src/       : C++/CUDA source code
- include/   : header files
- data/      : CIFAR-10 dataset (NOT committed, downloaded at runtime)
- models/    : saved weights / features (NOT committed)
