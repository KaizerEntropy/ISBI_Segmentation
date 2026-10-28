<div align="center">

# **A Dual-Branch Wavelet-ResNet Model with Cross-Attention and Sugeno Fuzzy Fusion for Nuclear Instance Segmentation**

### **IEEE International Symposium on Biomedical Imaging (IEEE ISBI)**
#### **Under Review &bull; IEEE Signal Processing Society &bull; IEEE Engineering in Medicine and Biology Society**

</div>

<p align="center">
  <a href="https://biomedicalimaging.org/"><img src="https://img.shields.io/badge/Conference-IEEE%20ISBI-0052cc.svg?style=flat" alt="IEEE ISBI"></a>
  <a href="https://biomedicalimaging.org/"><img src="https://img.shields.io/badge/Status-Under%20Review-FF6F00.svg?style=flat" alt="Status: Under Review"></a>
  <a href="https://signalprocessingsociety.org/"><img src="https://img.shields.io/badge/Sponsor-IEEE%20SPS%20%2F%20EMBS-003366.svg?style=flat" alt="IEEE SPS / EMBS"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat" alt="License: Apache 2.0"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=flat" alt="Python"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-11.8%2B-76B900.svg?style=flat" alt="CUDA"></a>
</p>

Official repository for **"A Dual-Branch Wavelet-ResNet Model with Cross-Attention and Fuzzy Fusion for Nuclear Segmentation and Instance Segmentation in Histopathology Images"**, submitted to the **[IEEE International Symposium on Biomedical Imaging (IEEE ISBI)](https://biomedicalimaging.org/)**.

**Authors:** Diptarko Bhattacharjee<sup>1,*</sup>, Mukut Chatterjee<sup>1,*</sup>, Dmitrij Kaplun<sup>2,3,✉</sup>, Ram Sarkar<sup>1</sup>  
<sup>1</sup> Department of Computer Science & Engineering, Jadavpur University, Kolkata, India  
<sup>2</sup> School of Computer Science and Technology / School of Artificial Intelligence, China University of Mining and Technology, Xuzhou, China  
<sup>3</sup> Intelligent Devices Institute, St. Petersburg Electrotechnical University "LETI", St. Petersburg, Russia  

<sup>*</sup> Equal contribution (co-first authors)  
<sup>✉</sup> Corresponding author: [dikaplun@etu.ru](mailto:dikaplun@etu.ru)  
**Conference Website:** [biomedicalimaging.org](https://biomedicalimaging.org/)

---

## Overview

Accurate instance segmentation of cell nuclei in routine hematoxylin and eosin (H&E) histopathology images is foundational for computational pathology workflows, guiding tumor grading, cellular pleomorphism quantification, and spatial micro-environment profiling. In high-grade malignant neoplasms, delineation is heavily compromised by tightly clustered nuclei, indistinct chromatin boundaries, and severe chromatic variability across laboratories.

Conventional convolutional networks (such as U-Net) rely on aggressive spatial downsampling via strided convolutions or max-pooling, which permanently eradicates high-frequency sub-band details and leads to under-segmentation along abutting membranes. While multi-task architectures like HoVer-Net regress distance-to-centroid fields, their intermediate pooling layers still suffer spatial aliasing. Vision transformers capture global dependencies but incur quadratic token costs while blurring sub-pixel cellular perimeters.

This framework couples multi-frequency wavelet analysis with deep residual representations through a symmetric dual-stream architecture:
- **Learnable Lifting Wavelet Transform (LLWT) Stream:** Decomposes features into multi-directional high-frequency detail sub-bands ($d_h, d_v, d_d$) via an algebraically invertible 2D lifting scheme with stain-adaptive spatial calibration, completely bypassing pooling loss and routing edge primitives directly to an Inverse LLWT decoder.
- **Residual Context Stream with GeM Pooling:** Extracts deep histological abstractions via pre-activation ResBlocks and learnable Generalized Mean (GeM) pooling, dynamically interpolating between average and max operations.
- **Multi-Scale Bi-Directional Cross-Attention (BiCA):** Exchanges contextual and geometric cues in channel-covariance space ($C \times C$) across 8 stages, utilizing zero-initialized Batch Normalization to preserve pre-trained feature stability during early training.
- **Skip-Connection Feature Alignment ($\mathcal{L}_{\text{feat}}$):** Supervised via channel-normalized Mean Squared Error, enforcing cosine similarity between upsampled decoder representations and encoder skip features to eliminate latent subspace divergence.
- **Learnable Sugeno Fuzzy Integral Decision Fusion:** Dynamically arbitrates between stream probabilities pixel-by-pixel using learned singleton fuzzy densities ($g_1, g_2$), resolving boundary ambiguities through non-additive fuzzy measures.
- **Multi-Task HoVer Head & Marker Watershed:** Regresses continuous horizontal and vertical distance-to-centroid vectors, yielding topological distance surfaces whose gradient basins drive marker-controlled watershed flooding.

---

## Model Architecture

<p align="center">
  <img src="figures/fig_architecture.pdf" alt="Dual-Branch Wavelet-ResNet Architecture" width="95%"/>
</p>

*Figure 1: Comprehensive architectural blueprint of the Dual-Branch Wavelet-ResNet network. The LLWT encoder decomposes features into low-frequency approximations and directional detail sub-bands $(d_h, d_v, d_d)$, which bypass deep abstractions directly to the Inverse LLWT decoder. Multi-scale Bi-Directional Cross-Attention (BiCA) synchronizes representations across both streams. Fused intermediate features feed dual heads for Sugeno semantic segmentation and HoVer distance regression, driving marker-controlled watershed instance decoding.*

### Structural Breakdown

| Stage / Component | Spatial Size | Channels (Wavelet / ResNet) | Primary Operations & Functional Description |
|:---|:---:|:---:|:---|
| **Input Tile** | $256 \times 256$ | 3 | Native-resolution 3-channel H&E histology image patch |
| **Stage 1 (E1 / D1)** | $128 \times 128$ | 128 / 64 | Polyphase LLWT splitting $\leftrightarrow$ BiCA-1 $\leftrightarrow$ ResNet Stem + GeM Downsampling |
| **Stage 2 (E2 / D2)** | $64 \times 64$ | 256 / 128 | Directional Predict/Update $\leftrightarrow$ BiCA-2 $\leftrightarrow$ PreAct ResBlocks; Auxiliary Head on D2 |
| **Stage 3 (E3 / D3)** | $32 \times 32$ | 512 / 256 | Multi-frequency Decomposition $\leftrightarrow$ BiCA-3 $\leftrightarrow$ Contextual ResBlocks; Auxiliary Head on D3 |
| **Stage 4 (E4 / D4)** | $16 \times 16$ | 512 / 512 | Deep Feature Extraction $\leftrightarrow$ BiCA-4 $\leftrightarrow$ Deep Semantic Subspace Alignment ($\mathcal{L}_{\text{feat}}$) |
| **Bottleneck** | $16 \times 16$ | 512 / 512 | Dual ASPP Bottlenecks (Dilation Rates: 1, 4, 7, Global Pooling) with 2D Spatial Dropout |
| **Heads & Fusion** | $256 \times 256$ | 1 (Seg) / 2 (HV) | Sugeno Fuzzy Integral ($p_{\text{fused}}$) $\rightarrow$ Mask; HoVer Head ($[-1, 1]$ Maps) $\rightarrow$ Watershed |

---

## Quantitative Benchmarks

The framework was evaluated across three public histopathology benchmarks: **MoNuSeg** (Multi-Organ Nuclei Segmentation, 7 organs), **TNBC** (Triple Negative Breast Cancer, pleomorphic malignant carcinoma), and **CPM-15** (Computational Pathology Challenge, glioblastoma, lung, and head/neck carcinoma). Performance is quantified via semantic overlap (**Dice**) and instance-level separation metrics (**Aggregated Jaccard Index [AJI]** and **Panoptic Quality [PQ]**).

### Performance Summary (Scores in %)

| Benchmark Dataset | Evaluation Metric | Prior State-of-the-Art | DualBranchNet (Ours) | Relative Margin |
|:---|:---:|:---:|:---:|:---:|
| **MoNuSeg** | Dice / AJI / PQ | 81.01 / 57.31 / 56.88 *(Trans-UNet)*<br>82.30 / 61.80 / 60.50 *(HoVer-Net)* | **82.46 / 62.44 / 61.49** | **+1.45 / +5.13 / +4.61**<br>*(+0.16 / +0.64 / +0.99 vs HoVer-Net)* |
| **TNBC** | Dice / AJI / PQ | 86.78 / 63.15 / 65.16 *(Trans-UNet)*<br>86.80 / 65.50 / 64.20 *(HoVer-Net)* | **88.00 / 70.23 / 69.69** | **+1.22 / +7.08 / +4.53**<br>*(+1.20 / +4.73 / +5.49 vs HoVer-Net)* |
| **CPM-15** | Dice / AJI / PQ | 88.45 / 68.78 / 63.17 *(Trans-UNet)*<br>87.78 / 67.34 / 62.39 *(U$^2$-Net)* | **90.02 / 70.82 / 67.40** | **+1.57 / +2.04 / +4.23** |

<details>
<summary><b>View complete comparative baselines across published architectures</b></summary>

| Method | Benchmark | MoNuSeg (Dice / AJI / PQ) | TNBC (Dice / AJI / PQ) | CPM-15 (Dice / AJI / PQ) |
|:---|:---|:---:|:---:|:---:|
| BoNuS (2023) | Boundary-Oriented CNN | -- / -- / -- | 76.73 / 60.73 / -- | 79.99 / 54.54 / -- |
| Swin-UNet (2022) | Hierarchical Vision Transformer | 80.12 / 53.12 / 54.87 | 84.71 / 61.46 / 63.12 | 86.92 / 65.79 / 61.28 |
| DELTASEG (2023) | Differential Boundary Network | 81.42 / -- / -- | 82.95 / -- / -- | -- / -- / -- |
| Trans-UNet (2021) | Hybrid CNN-Transformer | 81.01 / 57.31 / 56.88 | 86.78 / 63.15 / 65.16 | 88.45 / 68.78 / 63.17 |
| U$^2$-Net (2020) | Nested U-Structure | 80.92 / 56.72 / 56.02 | 85.92 / 62.01 / 62.58 | 87.78 / 67.34 / 62.39 |
| HoVer-Net (2019) | Distance Map Regression | 82.30 / 61.80 / 60.50 | 86.80 / 65.50 / 64.20 | 88.10 / 67.90 / 63.50 |
| **DualBranchNet (Ours)** | **Wavelet-ResNet Fuzzy Hybrid** | **82.46 / 62.44 / 61.49** | **88.00 / 70.23 / 69.69** | **90.02 / 70.82 / 67.40** |

</details>

---

## Qualitative Visualizations

<p align="center">
  <img src="figures/qualitative_outputs.png" alt="Qualitative Segmentation Outputs" width="95%"/>
</p>

*Figure 2: Multi-stage visual outputs produced by the system across representative test tissue sections. Columns (left to right): (1) Raw H&E input patch at native resolution, (2) Certified expert pathologist ground truth, (3) Sugeno fuzzy integrated continuous foreground probability map $\hat{p}_{\text{fused}}$, (4) Regressed horizontal and vertical HoVer distance maps showing distinct centroid-directed gradients, and (5) Marker-controlled watershed instance segmentation resolving dense cell clusters.*

---

## Ablation Studies and Analysis

Systematic modular ablations validate the mathematical contribution of each architectural block:

1. **Learnable Directional Lifting Preserves Fine Boundaries (+5.41% AJI on MoNuSeg):** Replacing fixed Haar wavelets with our learnable directional predict/update filters yields substantial gains (+3.48% Dice, +5.41% AJI on MoNuSeg). This confirms that learning spatial orientation vectors preserves sub-pixel nuclear margins that are permanently erased by conventional pooling.
2. **Sugeno Fuzzy Fusion Arbitrates Source Reliability (+2.32% AJI over Standard Sugeno, +3.23% over Concatenation):** Fusing branch probabilities via learned singleton densities ($g_1, g_2$) significantly outperforms linear averaging, concatenation, and gated attention. The fuzzy integral acts as an adaptive confidence gate, defaulting to the wavelet stream along sharp boundaries and the ResNet stream within homogeneous cell bodies.
3. **Skip Feature Alignment ($\mathcal{L}_{\text{feat}}$) Prevents Representation Drift:** Directing transposed-convolution upsamplings to align with encoder skip features via normalized MSE maximizes cosine similarity ($2(1 - \cos\theta)$), providing steady convergence gains (+1.55% Dice, +2.25% AJI on MoNuSeg).
4. **Mean Squared Gradient Error (MSGE) Enforces Steep Topographic Basins (+5.0 AJI Gain):** Supervising horizontal and vertical Sobel gradients across distance maps suppresses artifacts on the topological energy surface, preventing spurious over-segmentation during watershed flooding.

<details>
<summary><b>View official modular and loss ablation tables</b></summary>

### Modular Architecture Ablation (Scores in %)
| Component Under Test | Tested Configuration | MoNuSeg (Dice / AJI / PQ) | TNBC (Dice / AJI / PQ) | CPM-15 (Dice / AJI / PQ) |
|:---|:---|:---:|:---:|:---:|
| **ResUNet Backbone** | EfficientNet-B0 Block | 80.66 / 57.91 / 58.87 | 86.10 / 67.18 / 62.95 | 87.89 / 67.61 / 63.45 |
| | VGG-16 Block | 80.59 / 57.27 / 56.11 | 86.78 / 62.95 / 61.73 | 87.38 / 67.71 / 63.14 |
| | DenseNet-121 Block | 80.90 / 58.82 / 58.27 | 86.77 / 67.98 / 65.30 | 88.98 / 66.88 / 61.71 |
| | **ResBlock (Ours)** | **80.99 / 59.04 / 58.19** | **87.08 / 68.62 / 65.69** | **89.46 / 70.38 / 64.94** |
| **Wavelet Scheme** | Fixed Haar Wavelet | 80.52 / 60.95 / 58.96 | 86.09 / 61.36 / 61.97 | 88.30 / 67.58 / 65.81 |
| | Single Shared MLP Predict | 80.87 / 57.02 / 57.65 | 86.97 / 63.52 / 62.72 | 88.24 / 65.77 / 63.21 |
| | **Directional LLWT (Ours)** | **80.95 / 60.36 / 59.92** | **87.23 / 67.18 / 66.87** | **89.11 / 68.58 / 65.71** |
| **Decision Fusion** | Feature Concatenation | 80.78 / 59.21 / 58.93 | 84.62 / 62.01 / 62.91 | 85.70 / 67.45 / 64.34 |
| | Majority Voting | 80.41 / 58.75 / 57.21 | 85.45 / 61.97 / 63.02 | 87.34 / 67.62 / 62.03 |
| | Standard Fixed Sugeno | 81.56 / 60.12 / 59.78 | 87.01 / 63.00 / 64.72 | 88.98 / 69.31 / 63.49 |
| | **Learnable Sugeno (Ours)**| **82.46 / 62.44 / 61.49** | **88.00 / 70.23 / 69.69** | **90.02 / 70.82 / 67.40** |

### Composite Loss Objective Progression (Scores in %)
| $\mathcal{L}_{\text{seg}}$ | $\mathcal{L}_{\text{feat}}$ | $\mathcal{L}_{\text{hv}}$ | $\mathcal{L}_{\text{bnd}}$ | MoNuSeg (Dice / AJI / PQ) | TNBC (Dice / AJI / PQ) | CPM-15 (Dice / AJI / PQ) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\checkmark$ | -- | -- | -- | 77.65 / 53.45 / 53.15 | 84.15 / 61.40 / 61.15 | 85.30 / 61.80 / 59.35 |
| $\checkmark$ | $\checkmark$ | -- | -- | 79.20 / 55.70 / 55.15 | 85.42 / 63.60 / 63.10 | 86.95 / 64.25 / 61.05 |
| $\checkmark$ | $\checkmark$ | $\checkmark$ | -- | 81.15 / 60.70 / 59.75 | 87.10 / 68.60 / 67.85 | 89.15 / 69.10 / 65.25 |
| $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | **82.46 / 62.44 / 61.49** | **88.00 / 70.23 / 69.69** | **90.02 / 70.82 / 67.40** |

</details>

---

## Method Details

<details>
<summary><b>Mathematical formulation and inference pipeline</b></summary>

### 1. Invertible 2D Learnable Lifting Wavelet Transform
Feature tensor $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ is split into four polyphase grids: reference ($x_{00}$), horizontal ($x_{01}$), vertical ($x_{10}$), and diagonal ($x_{11}$). 
- **StainAdapt Calibration:** Global spatial descriptor $\mathbf{y}_{\text{gap}} = \operatorname{GAP}(x_{00})$ scales polyphase features via bottleneck MLP: $S(x_{00}, f) = f \odot \sigma(\mathbf{W}_2 \operatorname{ReLU}(\mathbf{W}_1 \mathbf{y}_{\text{gap}} + \mathbf{b}_1) + \mathbf{b}_2)$.
- **Directional Predict:** $d_k = x_k - S_k(x_{00}, \mathcal{P}_k(x_{00}))$ for $k \in \{01, 10, 11\}$ with depthwise convolutions $\mathcal{P}_k$.
- **Directional Update:** $c_l = x_{00} + \sum_{k} k_k \mathcal{U}_k(d_k)$ with depthwise operators $\mathcal{U}_k$ and learned scalars $k_k$.
- **Exact Invertibility (Theorem 1):** Because update and predict operations form a triangular uncoupled system, the downsampling is mathematically lossless:
  $$x_{00} = c_l - \sum_{k} k_k \mathcal{U}_k(d_k), \qquad x_k = d_k + S_k(x_{00}, \mathcal{P}_k(x_{00}))$$

### 2. Multi-Scale Bi-Directional Cross-Attention (BiCA)
To eliminate the quadratic $(HW)^2$ memory bottleneck of spatial self-attention, BiCA flattens spatial dimensions into $N = HW$ tokens and computes cross-covariance in channel space:
$$\mathbf{A}_{i, j} = \frac{1}{\sqrt{N}} \langle \mathbf{Q}_{i, :}, \, \mathbf{K}_{j, :} \rangle, \qquad \tilde{\mathbf{A}} = \operatorname{Softmax}_{\text{row}}(\mathbf{A})$$
$$\hat{\mathbf{x}}_q = \mathbf{x}_q + \mathbf{W}_{\text{out}}(\tilde{\mathbf{A}}\mathbf{V})$$
- **Zero-Initialized Identity Mapping (Theorem 2):** Weight $\gamma$ and bias $\beta$ of Batch Normalization in $\mathbf{W}_{\text{out}}$ are initialized to zero ($\gamma = 0, \beta = 0$), ensuring $\hat{\mathbf{x}}_q^{(0)} \equiv \mathbf{x}_q^{(0)}$ at step $t=0$, insulating pre-trained ResNet features from random early gradient shocks.

### 3. Sugeno Fuzzy Integral Decision Fusion
For evidence probabilities $p_w, p_r \in [0, 1]$ and learned singleton densities $g_1, g_2 = \sigma(\theta_1), \sigma(\theta_2)$, the Sugeno fuzzy integral evaluates to the closed-form expression:
$$p_{\text{fused}} = \max\Big( \min\big(\max(p_w, p_r), \, g_{\max}\big), \, \min(p_w, p_r) \Big)$$
where $g_{\max} = g_1$ if $p_w \ge p_r$, else $g_2$. The fused probability is converted to logit space via the clamped inverse sigmoid for numerically stable BCE loss computation:
$$\mathbf{z}_{\text{fused}} = \ln(p_{\text{fused}} + \epsilon) - \ln(1 - p_{\text{fused}} + \epsilon)$$

### 4. Composite Multi-Task Training Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_f \mathcal{L}_{\text{feat}} + \lambda_{hv} \mathcal{L}_{hv} + \lambda_b \mathcal{L}_{\text{boundary}}$$
- **Segmentation Loss ($\mathcal{L}_{\text{seg}}$):** Blends Focal Tversky ($\alpha=0.7, \beta=0.3, \gamma=0.75$) with dynamically weighted Binary Cross-Entropy across primary and intermediate auxiliary decoders (D3, D2).
- **Skip Alignment Loss ($\mathcal{L}_{\text{feat}}$):** Minimizes normalized MSE across all 4 stages: $\sum_{l=1}^4 \|\frac{\mathbf{u}_l^r}{\|\mathbf{u}_l^r\|_2} - \frac{\mathbf{s}_l^r}{\|\mathbf{s}_l^r\|_2}\|_2^2 \equiv 2(1 - \cos\theta)$.
- **HoVer Distance Loss ($\mathcal{L}_{hv}$):** Combines global MSE on $[-1, 1]$ normalized centroid offsets with foreground-restricted Mean Squared Gradient Error (MSGE) using horizontal/vertical Sobel kernels.
- **Boundary Consistency Loss ($\mathcal{L}_{\text{boundary}}$):** Penalizes $L_1$ discrepancy between Sobel gradient magnitudes: $\| \, |\nabla_S \sigma(\mathbf{z})| - |\nabla_S \mathbf{M}_{\text{gt}}| \, \|_1$.

### 5. Seamless Sliding-Window Blending and Marker-Controlled Watershed
- **2D Raised-Cosine Stitching:** Inference applies a Hanning window kernel $\mathbf{W}(i, j) = \max(w(i), 0.1) \cdot \max(w(j), 0.1)$ across overlapping patches (stride 192, crop size 256) to eliminate boundary tiling seams.
- **Topological Energy Field:** Constructed from normalized Sobel gradient magnitudes on horizontal and vertical distance maps:
  $$\tilde{\mathbf{G}} = \frac{\mathbf{G} - \min(\mathbf{G})}{\max(\mathbf{G}) - \min(\mathbf{G}) + \epsilon}, \qquad \mathbf{G} = \max(|\nabla \hat{\mathbf{H}}|, \, |\nabla \hat{\mathbf{V}}|)$$
- **Marker Flooding:** Foreground seeds identified where $\tilde{\mathbf{G}} < \tau_{\text{seed}}$ are eroded by disk radius $r_{\text{erode}}$ and flooded across $\tilde{\mathbf{G}}$ bounded strictly within the binary mask $\hat{p}_{\text{fused}} \ge 0.5$. Artifacts smaller than 10 pixels are removed.
</details>

---

## Citation

If you use this work, codebase, or results in your research, please cite our paper:

```bibtex
@inproceedings{bhattacharjee2025dualbranch,
  title     = {A Dual-Branch Wavelet-ResNet Model with Cross-Attention and Fuzzy Fusion for Nuclear Segmentation and Instance Segmentation in Histopathology Images},
  author    = {Bhattacharjee, Diptarko and Chatterjee, Mukut and Kaplun, Dmitrij and Sarkar, Ram},
  booktitle = {Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2025},
  organization = {IEEE},
  url       = {https://biomedicalimaging.org/}
}
```

---

## Acknowledgements

Computational resources were provided by the Centre for Microprocessor Applications for Training Education and Research (CMATER), Department of Computer Science & Engineering, Jadavpur University, Kolkata, India. Part of this research was supported by the Intelligent Devices Institute, St. Petersburg Electrotechnical University "LETI", Russia.

---

## License and IEEE Terms

This project is distributed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete details.

In accordance with IEEE conference publication policies and author guidelines, please review the IEEE Intellectual Property Rights and symposium documentation at the [IEEE International Symposium on Biomedical Imaging (ISBI)](https://biomedicalimaging.org/) portal and the [IEEE Open Access Policies](https://www.ieee.org/publications/rights/index.html).

---

## Contact

For inquiries regarding the methodology, mathematical formulations, or reproduction of experimental benchmarks:

- **Mukut Chatterjee** (Co-First Author): [mukutchatterjee0211@gmail.com](mailto:mukutchatterjee0211@gmail.com) — Jadavpur University, Kolkata, India
- **Diptarko Bhattacharjee** (Co-First Author): [diptarko.1312@gmail.com](mailto:diptarko.1312@gmail.com) — Jadavpur University, Kolkata, India
- **Dmitrij Kaplun** (Corresponding Author): [dikaplun@etu.ru](mailto:dikaplun@etu.ru) — ETU "LETI", St. Petersburg, Russia
- **Symposium Portal:** [IEEE International Symposium on Biomedical Imaging (ISBI)](https://biomedicalimaging.org/)
