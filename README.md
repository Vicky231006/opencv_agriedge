# AgriEdge: Multi-Scale Transformer Segmentation & Dense Spatial Risk Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-00FF66.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-EE4C2C.svg?style=flat-square)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue.svg?style=flat-square)](https://huggingface.co/)

An enterprise-grade computer vision and biometrics pipeline designed to execute pixel-level semantic crop lesion segmentation. Moving past passive "Yes/No" or bounding-box classification, this system extracts fine-grained edge matrices from high-resolution leaf canopies and maps active structural anomalies directly onto a continuous agricultural yield risk decay curve.
---
## 🚀 Live Demonstration

![Walkthrough](recording.gif)

---

## 🚀 Architectural Blueprint

The architecture implements a highly decoupled **Encoder-Decoder-Regression** pipeline optimized for high-fidelity spatial reasoning without structural coupling. 

### 1. Multi-Scale Attention Encoder (Backbone)
The vision pipeline utilizes an attention-driven **Hierarchical Vision Transformer (SegFormer MiT-B0)** backbone. Unlike standard convolutional architectures (e.g., EfficientNet) that apply a localized receptive field, the Self-Attention mechanism preserves global structural context across multi-scale dimensions. The encoder processes input resolutions at $224 \times 224 \times 3$ and extracts four multi-scale hidden state tensors:
* **`c1`**: $[B, 32, 56, 56]$ (High-resolution structural edge features)
* **`c2`**: $[B, 64, 28, 28]$ (Intermediate textural groupings)
* **`c3`**: $[B, 160, 14, 14]$ (Localized anomaly contexts)
* **`c4`**: $[B, 256, 7, 7]$ (Deep global semantic feature embeddings)

### 2. Pyramidal Feature Fusion Decoder (Custom Neck & Head)
The extracted multi-scale representations are fed into a custom `SegFormerDecoderHead`. The decoder uses an MLP infrastructure to project the varying channel dimensions uniformly down to $128$ channels. Features are upsampled via bilinear interpolation to match the spatial dimensions of `c1` ($56 \times 56$), concatenated along the channel axis ($128 \times 4 = 512$), and fused via a $1 \times 1$ 2D Convolution block. 

To resolve localized label mismatches, the final logits pass through a secondary high-fidelity bilinear scaling matrix, expanding the spatial array back to the native $224 \times 224$ footprint matching the raw ground-truth mask coordinates.

### 3. Continuous Financial Yield Risk Core
Instead of simple categorical labeling, the final binary argmax segmentation mask array ($224 \times 224$) is streamed straight into a non-linear regression block. The engine isolates lesion pixels ($Class\ 1$) against healthy and background pixels ($Class\ 0$) to compute an exact mathematical surface-area degradation curve.

---

## 🧠 Mathematical Formulation

### 1. Inverse Class-Weighted Cross-Entropy
To resolve severe spatial class imbalances (where healthy background pixels occupy over 95% of an image canvas and newly forming fungal streaks occupy less than 5%), optimization gradients are scaled dynamically using inverse frequency ratios. This forces the optimization path to penalize segmentation failures on rare lesion points significantly harder:

$$\mathcal{L}_{\text{WeightedCE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{1} w_c \cdot y_{i,c} \log(\hat{y}_{i,c})$$

Where penalties are structurally distributed across class indices:
* $w_{0}\ (\text{Background}) = 1.0$
* $w_{1}\ (\text{Active Disease Lesions}) = 10.0$

### 2. Non-Linear Exponential Yield Decay
Real-world agricultural crop degradation does not scale linearly. Beyond a specific spatial infestation threshold, localized tissue decay compromises surrounding cellular infrastructure exponentially. The `YieldRiskCore` calculates this degradation via an integrated exponential area matrix:

$$\text{Estimated Yield Loss } (\%) = \left(1.0 - e^{-\kappa \cdot \left(\frac{\text{Area}_{\text{Lesion}}}{\text{Area}_{\text{Total Leaf Surface}}}\right)}\right) \times 100$$

Where $\kappa = 3.5$ represents the disease virulence attenuation factor, outputting a precise continuous risk score (e.g., `35.16% Yield Loss at Risk`) for actionable supply-chain automation.

---

## 📦 Data Engine Specification

* **Ingestion Source:** Roboflow / Kaggle Augmented Leaf Disease Segmentation Framework (~7,056 multi-channel arrays).
* **Input Target Tensor Geometry:** `[Batch, 3, 224, 224]` (Float32 normalized RGB arrays).
* **Target Target Mask Geometry:** `[Batch, 224, 224]` (Long64 single-channel index matrices).
* **Compression Resolution:** Due to JPEG compression artifacting, masks are passed through an explicit channel intensity mask filter. Pixels are registered as active lesions when matching a uniform value profile of exactly `38` across all three image dimensions.

---
## Model Performance Evaluation

During training execution, optimization convergence profiles verify an active learning state. The model breaks away from zero-prediction shortcuts within the first 20 mini-batches, converging cleanly down from an initial loss profile of 0.28 to a steady 0.13 inside 5 epochs.

```markdown
🚀 Launching 5-Epoch Optimization Cycle...
Epoch [1/5] -> Loss Profile: 0.2760 | Processed in 45.2s
Epoch [2/5] -> Loss Profile: 0.1634 | Processed in 44.8s
Epoch [3/5] -> Loss Profile: 0.1558 | Processed in 44.9s
Epoch [4/5] -> Loss Profile: 0.1511 | Processed in 45.1s
Epoch [5/5] -> Loss Profile: 0.1325 | Processed in 44.7s
✅ Model Convergence Cycle Finalized.
```