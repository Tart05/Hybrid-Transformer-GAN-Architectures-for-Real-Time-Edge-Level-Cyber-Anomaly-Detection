# Hybrid-Transformer-GAN-Architectures-for-Real-Time-Edge-Level-Cyber-Anomaly-Detection
Baseline hybrid model for anamoly detection

# Baseline Transformer-GAN for Cyber Anomaly Detection

This repository contains the baseline implementation of a Hybrid Transformer-GAN architecture designed for detecting anomalies in IoT network traffic (MQTT). This model acts as the "Teacher" in a Knowledge Distillation pipeline.

## 🚀 Overview
The model utilizes a **Transformer-Encoder** as a Generator to learn the latent distribution of normal network packets. It identifies anomalies by calculating the reconstruction error; packets that cannot be accurately reconstructed are flagged as potential threats.

## 🛠️ Architecture
- **Generator:** 4-6 Layer Transformer Encoder with Multi-Head Self-Attention.
- **Discriminator:** Multi-layer Perceptron (MLP) with Leaky ReLU activations.
- **Detection Logic:** Reconstruction Error Thresholding.

## 📦 Setup & Installation
1. **Environment:**
   ```bash
   conda create -n baseline_gan python=3.10
   conda activate baseline_gan
   pip install torch pandas scikit-learn numpy
---
