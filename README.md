# Deep-Learning-Based-Spectrum-Sensing-using-CNN-GAP
Deep Learning‑Based Spectrum Sensing using CNN and GAP Layers
This project implements a complete binary spectrum sensing system capable of identifying whether the received I/Q signal frame contains:

Primary User (PU) signal
Noise only (no PU)

The system uses 1D CNN architectures, feature‑optimized layers, and robust dataset filtering to achieve highly accurate detection under various SNR conditions.

📌 Overview
This work focuses on building a reliable spectrum sensing pipeline using deep‑learning CNN models trained on raw 1024‑sample I/Q signals.
The goal is to create a fast, lightweight, high‑accuracy model capable of generalizing across modulation types and noisy environments.

🧩 Signal Processing & Feature Handling
The system processes radio signals using:
🔹 I/Q Raw Input Handling
Utilizes the in‑phase and quadrature (I/Q) samples provided by the dataset.
🔹 AWGN Noise Augmentation
Adds controlled‑standard‑deviation noise (σ = 0.708) dynamically during training.
🔹 HDF5 Data Pipeline
Efficient sample loading, sorted indexing, and batched generators for high‑speed training.
🔹 SNR Filtering
Training uses only SNR ≥ –8 dB, eliminating noise‑dominated frames that harm accuracy.

⚙️ Model Architecture
The project implements two CNN architectures:
🔹 CNN + Flatten Layer

Multiple Conv1D + MaxPooling layers
Dense classifier
High accuracy but prone to overfitting

🔹 CNN + Global Average Pooling (GAP)

Replaces flatten layer with GAP
Reduces parameters drastically
Improves generalization
Enhances detection performance across modulation types

Both models output a binary classification:
Signal present (1) / Noise (0)

🤖 Training Setup

Trained on filtered RadioML dataset (I/Q: 1024×2 samples)
Uses Adam optimizer
Batch size: 2048
Epochs: 32–40 depending on configuration
HDF5‑based generator ensures memory‑safe training
ModelCheckpoint + ReduceLROnPlateau callbacks included


📊 Evaluation Metrics
Each model is evaluated using:

Accuracy
Loss curves (train/val)
Detection Probability (Pd) vs SNR
Performance across modulation types
Comparison with traditional sensing methods

The system consistently outperforms:

Maximum–Minimum Eigenvalue Ratio method
Frequency‑Domain Entropy detection


🔍 SNR‑Based Performance Analysis
Evaluation includes:

Pd curves for multiple modulations (e.g., OOK, QPSK)
Behavior under extremely low SNR conditions
Identification of SNR thresholds where PU presence becomes detectable

The model achieves:

97.5%–98% detection accuracy depending on architecture
Strong robustness against AWGN
Excellent generalization across unseen modulations


📁 Automated Plotting & Visualization
The system automatically generates:

Training accuracy curves
Training loss curves
Final Pd vs SNR plots
Modulation‑wise performance graphs

Plots are saved to the configured output directory.

📂 Project Structure
/src
    data_loader.py
    gap_model.py
    flatten_model.py
    generator.py
    train.py
    evaluate.py

/data
    (Place HDF5 dataset here)

plots/
checkpoints/
README.md


🚀 Future Enhancements

Hardware deployment on SDR platforms
Multi‑class spectrum classification
Hybrid CNN‑RNN architectures
Transfer learning for unseen modulation types
Real‑time inference optimization


📜 License
Open for academic and research usage.

👤 Author
Developed by Mariam Mohamed
