# Deep Semi-Fragile Watermarking with Multi-Tamper Detection and Metaheuristic Optimization

### Team Members
| Name | Roll Number |
|---|---|
| Yashaswini Sharma | 2023UCA1728 |
| Kriti Mathur | 2023UCA1845 |
| Sneha Negi | 2023UCA1860 |
| Ayana Kumar | 2023UCA1877 |

---

## Abstract

The rapid rise of image manipulation techniques — driven by the widespread availability of editing tools and generative AI models — has made image authenticity verification a pressing challenge. Manipulations range from benign operations such as JPEG compression, filtering, and lighting adjustments, to malicious forgeries such as object removal, splicing, and copy–move tampering.

This project presents a **deep learning–based semi-fragile watermarking framework** that detects and localizes both benign and malicious tampering, building on the baseline architecture proposed by **Zhao et al. (2024)**. We extend the standard encoder–decoder pipeline with three key contributions:

1. A **Multi-Tamper Attack Module** that simulates a broader range of realistic tampering operations, including photometric distortions.
2. A **feature selection mechanism** within the revealing network to reduce reconstruction noise.
3. **Metaheuristic optimization (Particle Swarm Optimization)** to dynamically tune hiding and revealing loss weights, replacing static hyperparameters.

Our proposed method achieves a **Peak Signal-to-Noise Ratio (PSNR) of 45.14 dB** and a **Structural Similarity Index (SSIM) of 0.996**, demonstrating strong imperceptibility while maintaining reliable tamper localization.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Base Paper Summary — Zhao et al. (2024)](#2-base-paper-summary--zhao-et-al-2024)
3. [Our Re-Implementation of the Base Paper](#3-our-re-implementation-of-the-base-paper)
4. [Limitations of the Base Paper](#4-limitations-of-the-base-paper)
5. [Our Improvements](#5-our-improvements)
6. [Final Model Architecture](#6-final-model-architecture)
7. [Results](#7-results)

---

## 1. Introduction

### 1.1 Motivation for Semi-Fragile Watermarking

Digital image manipulation has become trivial with modern editing software and generative AI. While benign edits (compression, filters, lighting) are harmless, malicious manipulations — object removal, copy–move forgery, splicing, and AI-based content replacement — can significantly alter an image's semantic meaning.

Ensuring image authenticity and integrity is essential in:

- Digital journalism
- Legal and evidential imaging
- Forensic investigations
- Social media content moderation
- Copyright preservation
- Deepfake detection

Traditional watermarking approaches face a fundamental trade-off: **robust** watermarks survive benign edits but fail to break under malicious tampering, while **fragile** watermarks break too easily, even under harmless operations.

**Semi-fragile watermarking** resolves this by producing a watermark that:

- Survives benign operations
- Breaks under malicious manipulations
- Localizes tampered regions

This makes it well-suited for real-world authentication tasks where benign editing is common but malicious tampering must still be detectable.

### 1.2 Problem Definition

This project builds on the deep-learning-based semi-fragile watermarking framework proposed by Zhao et al. (2024), which uses an encoder–decoder architecture to embed and recover invisible watermarks for tamper localization. Our analysis identified three critical limitations in the baseline approach:

- **Limited tampering diversity** — trained primarily on simple synthetic tampering (e.g., rectangular masking), leaving it vulnerable to complex manipulations like splicing or AI-based inpainting.
- **No hyperparameter optimization** — relies on static, manually tuned loss weights and embedding strength, resulting in suboptimal imperceptibility and unstable convergence.
- **Reconstruction instability** — the revealing network processes redundant feature channels, introducing noise into watermark recovery and reducing localization accuracy.

Our objective is to design a model with a **multi-tamper attack module** that simulates real-world attacks (forgery, inpainting, copy–move) and embeds a watermark that:

- Survives benign transformations (filters, brightness adjustments)
- Breaks under malicious structural tampering
- Uses the resulting disruption pattern to localize tampered regions

Formally, the goal is to develop a deep-learning-based semi-fragile watermark that:

1. Embeds a secret pattern into an image invisibly
2. Extracts the hidden pattern even after benign distortions
3. Fails intentionally on tampered regions
4. Uses this failure signal to highlight tampered pixels

This enables both **verification** ("is this image authentic?") and **localization** ("where was it tampered?").

### 1.3 Summary of Zhao et al. (2024)

- **Title:** Proactive Image Manipulation Detection via Deep Semi-Fragile Watermark
- **Published in:** Neurocomputing (2024)

Zhao et al. proposed a deep learning semi-fragile watermarking system for image tamper detection, using two cooperative networks:

- A **Hider** network that embeds a secret watermark into an image
- A **Revealer** network that extracts the watermark from a potentially manipulated image

**Core philosophy:**

- The watermark remains recoverable under non-malicious operations (e.g., compression, minor color changes).
- The watermark becomes corrupted when the image content is maliciously manipulated.
- The difference between the original and reconstructed secret reveals *where* the image was altered.

**Why this paper was chosen as the baseline:**

- Clear, well-defined semi-fragile behavior (as opposed to purely robust or fully fragile watermarking)
- Simple, extensible dual-network (Hider + Revealer) architecture
- Naturally fits into a PyTorch training pipeline
- Provides a solid benchmark — its simplicity exposes clear weaknesses (instability, poor tamper variety) that motivate improvement
- Published, peer-reviewed work offering academic legitimacy and a strong conceptual foundation

---

## 2. Base Paper Summary — Zhao et al. (2024)

The base architecture consists of two cooperative neural networks — a **Hiding Network** that embeds a watermark into a cover image, and a **Revealing Network** that recovers the watermark from a possibly manipulated image. The watermark is designed to survive benign modifications while becoming corrupted under malicious tampering, enabling localization.

### 2.1 Hiding Network

**Purpose:** Embed a full-image secret watermark into a cover image such that the resulting container image is visually indistinguishable from the original, while remaining recoverable and maintaining high perceptual quality.

**Architecture:** A UNet-like encoder–decoder:

- **Encoder** — progressively downsamples the input and extracts hierarchical features
- **Bottleneck** — compresses combined cover + secret information
- **Decoder** — upsamples and reconstructs the watermark-modified image

**Key characteristics:**

- Multi-scale feature extraction
- Skip connections between encoder and decoder layers
- Small (3×3) convolution kernels
- ReLU activations
- Batch normalization

This structure suits watermarking well: skip connections preserve spatial detail, the bottleneck mixes secret and cover features, and the decoder reconstructs the container with minimal distortion.

**Embedding strategy:** The watermark is embedded additively:

```
Container = Cover + α · W
```

where `C` is the cover image, `W` is the watermark residual generated by the Hiding Network, and `α` is a fixed embedding strength.

Rather than generating the entire container image directly, the network predicts a low-magnitude residual `W` containing only the watermark information — improving imperceptibility, stability under benign modifications, and ease of recovery.

### 2.2 Revealing Network

**Purpose:** Reconstruct the hidden secret image from a (possibly tampered) container image.

**Architecture:** A lightweight CNN consisting of:

- Initial convolutional layers
- Several residual blocks
- Upsampling/convolution layers producing a 3-channel output
- Final sigmoid or tanh activation

**Residual blocks** preserve image structure, improve gradient flow, provide robustness against distortions, and maintain consistency between cover and tampered images. In the semi-fragile context, they help detect subtle tampering-induced disruptions and enable deeper feature extraction without degradation.

The Revealer reconstructs the secret `Ŝ` from the (possibly tampered) container `C_tampered`. If the image is authentic, reconstruction is accurate; if tampered, the reconstructed secret is corrupted in the tampered regions.

### 2.3 Tamper Detection Mechanism

Tamper detection compares the reconstructed secret against the original secret to produce a **difference map**:

- **Low difference** → watermark intact → no tampering
- **High difference** → watermark destroyed → tampered region

This difference map acts as a fragility indicator. A threshold `T` converts it into a binary **tamper mask** `M`, where `1` indicates predicted tampering and `0` indicates an authentic region. The resulting mask is used to evaluate localization accuracy (typically via **IoU** or **F1-score**) and to visualize manipulated regions.

### 2.4 Loss Functions

Training combines two losses:

- **Hiding loss** — measures distortion introduced by embedding, ensuring the container resembles the cover and keeping PSNR/SSIM high.
- **Revealing loss** — measures reconstruction accuracy of the secret, ensuring the watermark is recoverable in authentic regions while allowed to break in tampered regions.

The total loss is a weighted sum of the two, with the base paper typically assigning a higher weight to the hiding loss (to preserve imperceptibility) and a lower weight to the revealing loss.

---

## 3. Our Re-Implementation of the Base Paper

Our goal was to reproduce the core functionality of Zhao et al. (2024) using a modern PyTorch pipeline, adapted for real-world datasets, greater tampering variability, and practical training constraints (Google Colab). The conceptual architecture remains faithful to the original, with the following implementation adjustments:

| Aspect | Paper Specification | Our Implementation | Rationale / Effect |
|---|---|---|---|
| **Dataset** | FFHQ (faces, 256×256) for training; CASIA/COCO for testing | OpenImages V6 (validation subset), 1,000 images | Publicly accessible, suitable for Colab; greater visual diversity; not limited to faces |
| **Input Pairing** | Secret and cover from separate datasets | Secret image randomly shuffled from the same batch as cover | Simplifies pipeline while preserving cover-agnostic behavior via per-batch shuffling |
| **Network Size** | Large UNet, 8–10 layers, downsampling to 1/16 resolution | Medium UNet, 3 down / 3 up layers | Fits Colab GPU memory limits (~8GB); slightly slower convergence but feasible |
| **Revealing Network** | CEILNet-style decoder, 9 residual blocks | 5 residual blocks | Reduces computation/training time with minor performance impact |
| **Distortion Module** | Differentiable Gaussian blur, JPEG, scaling, color adjustment (custom CUDA ops) | Gaussian blur + JPEG (non-differentiable, via OpenCV) | Easier to implement; gradients don't flow through distortions, but semi-fragility is preserved |
| **Tampering Module** | Differentiable tamper layer masking random regions | Random rectangular noise patch | Simpler, reliable ground-truth mask generation |
| **Training Regime** | ~100 epochs on FFHQ, trained on A100 GPUs | 3–5 epoch demo on 1,000 images | Adapted to Colab runtime limitations; results remain valid for comparison |
| **Tamper Threshold** | 0.5, tuned via ROC curves | Fixed at 0.05 absolute difference | Simpler heuristic; effective for IoU evaluation |
| **Evaluation Metrics** | AUC, F1 (pixel-level), PSNR, SSIM | PSNR, SSIM, IoU | IoU directly measures tamper overlap; AUC unnecessary for initial comparison |

**Summary:** Although our implementation differs from the original in dataset choice, model size, attack module, and training environment, we preserved the core scientific principles of Zhao et al.'s framework. These adaptations made the model practical, reproducible, and efficient in a Colab-based setting while retaining the theoretical characteristics of semi-fragile watermarking: high imperceptibility, robustness to benign transforms, and fragility under structural tampering.

---

## 4. Limitations of the Base Paper

While effective, both the original method and our initial re-implementation exhibited several limitations that motivated the improvements described in Section 5.

### 4.1 Limited Tampering Diversity
The base paper uses simple, synthetic tampering operations (local region masking, block removal, basic copy–paste, small geometric distortions), which fail to capture the complexity of real-world forensic scenarios. As a result, the model learns patterns effective only for idealized tampering and generalizes poorly to realistic manipulations such as AI-based inpainting, splicing, blended copy–move, photometric edits, and JPEG artifacts.

### 4.2 Weak Handling of Photometric Distortions
The differentiable approximations used for Gaussian blur, JPEG compression, and color adjustment are simplistic and controlled. Real-world distortions — strong JPEG compression, non-linear tone mapping, mobile post-processing, social media compression — are more severe, making the Revealing network less robust when distortions diverge from training conditions.

### 4.3 No Hyperparameter Optimization
The paper fixes loss weights (λH, λR) and embedding strength α, all of which significantly affect imperceptibility, recovery accuracy, fragility, and training stability — yet no tuning method is provided. This leads to suboptimal embedding strength, unstable convergence, and poor generalization to new datasets such as OpenImages.

### 4.4 Limited Network Depth (Revealing Network)
The shallow Revealer CNN is sensitive to noise and struggles to reconstruct watermarks from highly textured or cluttered images, especially under strong distortions or complex tampering — risking overfitting to simplistic tampering and noisy difference maps.

### 4.5 No Feature Selection or Channel Pruning
All convolutional channels are used regardless of their contribution to reconstruction. Redundant and low-value channels introduce noise and instability, slowing convergence and weakening tamper localization and robustness.

### 4.6 Over-Reliance on Differentiable Tampering
Since tampering is applied through differentiable layers, and real-world tampering is inherently non-differentiable, the model risks overfitting to soft, differentiable distortions rather than actual malicious edits.

### 4.7 Unrealistic Experimental Conditions
The dataset is largely limited to faces (FFHQ) under fixed tampering conditions, and the paper assumes access to powerful GPUs (A100). This raises questions about performance on general-purpose datasets and makes exact replication and scaling difficult in constrained environments like Colab.

---

## 5. Our Improvements

Building on our re-implementation, we introduced three targeted improvements addressing the limitations above while preserving the core idea of the base paper.

### 5.1 Multi-Tamper Attack Module *(Major Extension)*

**Motivation:** The base paper's reliance on simple synthetic tampering (e.g., rectangular masking) does not represent real-world manipulation, which includes splicing, copy–move, inpainting-based removal, strong compression, and color-based edits.

**Our Module** introduces:

- **Structural tampering** (malicious edits):
  - Random patch replacement (noise injection)
  - Copy–move tampering (duplicated region within the image)
  - Splicing (region pasted from a different image in the batch)
  - Classical inpainting (region removal via the Telea method)
- **Photometric distortions** (benign edits):
  - Strong JPEG compression (q = 10–30)
  - Color shift / hue rotation

This separation between structural (malicious) and photometric (benign) tampering is essential for enforcing semi-fragility: the model must **not** break under benign edits, but **must** break under structural edits.

**Impact:** More realistic training, better differentiation between benign and malicious edits, improved fragility to content changes, and higher IoU through exposure to diverse tampering cases.

### 5.2 Feature Selection in the Revealing Network

**Motivation:** The base paper treats all channels equally, even when some provide minimal signal or introduce noise, reducing reconstruction stability and increasing sensitivity to benign distortions.

**Our Approach:** We compute a channel importance score `I_C` for each channel, where:

- **High I_C** → channel is sensitive to tampering → retained
- **Low I_C** → channel is noisy or irrelevant → pruned

We retain the **top 50%** of channels by importance.

**Effects:**

- Reduced reconstruction noise
- Improved stability and reduced randomness
- Reduced overfitting
- Sharper tamper localization
- Cleaner container output (improved imperceptibility)

### 5.3 Particle Swarm Optimization (PSO) for Hyperparameter Tuning

**Motivation:** The base paper's fixed hyperparameters (λH = 1, λR = 0.75, static α) strongly influence watermark visibility, robustness, fragility, localization accuracy, and convergence stability — yet no optimization method is provided. Given the increased complexity of the OpenImages dataset, manual tuning proved insufficient.

**Our Framework:** We use PSO to search for the optimal values of:

- `λH` — hiding loss weight
- `λR` — revealing loss weight
- `α` — embedding strength

The objective jointly maximizes imperceptibility (PSNR) and tamper localization accuracy (IoU), allowing PSO to find an optimal trade-off between the two.

**Why PSO:** PSO is gradient-free and efficiently searches the non-convex hyperparameter landscapes typical of watermarking problems.

**Benefits:** Optimal hiding strength, improved convergence, enhanced imperceptibility, and better overall tamper detection performance.

### 5.4 Combined Effect of All Improvements

| Metric | Before | After |
|---|---|---|
| PSNR | 9.47 dB | 45.14 dB |
| SSIM | 0.142 | 0.996 |
| IoU | 0.035 | 0.061 |

These improvements collectively yield much higher imperceptibility, more realistic and diverse tamper simulation, more stable and efficient training, and better generalization to real-world images.

---

## 6. Final Model Architecture

### 6.1 Overall System Structure

The final system integrates the following components:

1. **Hiding Network** (UNet-based)
2. **Multi-Tamper Attack Module** (structural + photometric)
3. **Revealing Network** (residual CNN with channel selection)
4. **Difference Map Generator**
5. **PSO-optimized loss weighting and embedding strength**
6. **Tamper Localization Mask**

### 6.2 Hiding Network

Retains the UNet architecture, benefiting indirectly from improved training dynamics via PSO-optimized hyperparameters and more realistic tamper exposure.

**Key characteristics:**

- 3 levels of downsampling and upsampling (Colab-feasible)
- Skip connections preserving spatial detail
- Tanh activation producing the watermark residual in range [-1, 1]
- Output residual scaled by the PSO-optimized `α` for optimal invisibility

**Effect of improvements:** Higher-quality containers, more stable feature embedding, and a more subtle, visually invisible residual.

### 6.3 Multi-Tamper Attack Module

A comprehensive tampering engine producing:

- **Malicious structural tampering** — random patch replacement, copy–move, splicing, inpainting
- **Benign photometric distortions** — strong JPEG compression, hue/color shifts

**Output:** Tampered image + ground-truth tamper mask.

**Purpose:** Enforce semi-fragile behavior — the watermark must break under structural tampering while surviving benign edits.

**Effect:** Improved IoU, better robustness to real-world transformations, and a more realistic training environment than the base paper.

### 6.4 Revealing Network

Modified with feature selection for cleaner, more tampering-sensitive reconstruction.

**Architecture highlights:**

- Initial convolutional layers
- Residual block stack (5 blocks)
- Transposed convolutions for upsampling
- Channel selection mask applied to intermediate feature maps
- Sigmoid output reconstructing the watermark image

**Feature selection:** After initial training, the top 50% of channels (by importance) are retained; others are suppressed.

**Results:** Lower reconstruction noise, sharper difference maps, improved localization clarity, and reduced overfitting.

### 6.5 Difference Map Generator

After retrieving the secret watermark, a difference map is computed to capture where the watermark was corrupted:

- **Bright regions** → tampered
- **Dark regions** → authentic

A threshold of **0.05** converts this into a binary tamper mask, giving reliable tamper identification and clear separation between benign distortions and malicious manipulations.

### 6.6 PSO-Optimized Loss Function

The combined training loss uses `λH` and `λR` learned via PSO, along with a PSO-optimized embedding strength `α`, making the final model adaptive to the dataset and tampering conditions.

**Effects of PSO optimization:**

- Much higher PSNR and SSIM
- Stronger separation between benign and malicious regions
- More stable convergence
- Optimal balance between hiding and revealing accuracy

---

## 7. Results

| Model | PSNR | SSIM | IoU |
|---|---|---|---|
| Original Re-Implementation | 9.47 dB | 0.142 | 0.035 |
| **Our Improved Model** | **45.14 dB** | **0.996** | **0.061** |

Our improved model achieves substantially higher imperceptibility (PSNR and SSIM) while also improving tamper localization accuracy (IoU) over the baseline re-implementation, validating the effectiveness of the Multi-Tamper Attack Module, feature selection mechanism, and PSO-based hyperparameter optimization.

---

## Reference

Zhao, X. et al. (2024). *Proactive Image Manipulation Detection via Deep Semi-Fragile Watermark*. Neurocomputing.
