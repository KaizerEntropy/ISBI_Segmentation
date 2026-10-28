# Dual-Branch Wavelet-ResNet for Nuclear Instance Segmentation

Official PyTorch implementation of the dual-branch neural architecture combining Learnable Lifting Wavelet Transforms (LLWT), pre-trained ResNet representations, bi-directional cross-attention, and Sugeno fuzzy integral fusion for nuclear semantic and instance segmentation in histopathology images.

## Overview

Accurate delineation of cell nuclei in Hematoxylin and Eosin (H&E) stained tissue biopsies is essential for computational pathology workflows, including tumor grading, cellular pleomorphism quantification, and spatial tumor-infiltrating lymphocyte (TIL) profiling.

Standard convolutional architectures struggle with touching nuclei and suffer from pooling-induced boundary blurring, while vision transformers incur prohibitive computational costs and lack fine, sub-pixel edge representations. This framework resolves these challenges through an integrated dual-stream design:

1. Learnable Lifting Wavelet Transform (LLWT) Stream: Implements an algebraically invertible 2D lifting scheme with stain-adaptive spatial calibration to decompose features into directional high-frequency detail bands without pooling loss. Detail sub-bands bypass deep abstractions directly to an Inverse LLWT decoder.
2. Residual Semantic Stream: Utilizes pre-trained residual blocks with Generalized Mean (GeM) pooling to capture robust histological context.
3. Bi-Directional Cross-Attention (BiCA): Dynamically recalibrates channel-wise feature representations across streams with zero-initialized identity mapping for early training stability.
4. Skip-Connection Feature Alignment Loss (L_feat): Enforces directional alignment between upsampled decoder features and encoder skip features, eliminating representation divergence.
5. Sugeno Fuzzy Integral Fusion: Pixel-wise decision arbitration parameterized by learned singleton fuzzy densities (g1, g2), dynamically weighting stream confidences based on local boundary ambiguity.
6. Multi-Task HoVer Head: Regresses horizontal and vertical distance-to-centroid maps, driving marker-controlled watershed segmentation to cleanly separate abutting cell nuclei.

## Detailed Methodology and Mathematical Formulation

### 1. Mathematical Definition of Nuclear Instance Segmentation
Let I in R^(3 x H x W) represent an RGB histopathology image of height H and width W. Semantic segmentation estimates a binary mapping M in {0, 1}^(H x W) indicating foreground nuclear presence versus background stroma:

    M(y, x) = I( (y, x) in Union_{k=1}^K Omega_k )

where Omega_k denotes the compact spatial support of the k-th distinct cell nucleus, and K is the total nucleus count. 

Semantic segmentation cannot distinguish between abutting nuclei sharing a boundary: if partial Omega_i intersects partial Omega_j is non-empty, Omega_i and Omega_j merge into a single connected component. Standard 8-connected component labeling fails whenever dist(Omega_i, Omega_j) = 1. Instance segmentation resolves this by assigning a unique integer index to each independent nuclear instance:

    Y_inst(y, x) = k if (y, x) in Omega_k, else 0

### 2. Learnable Lifting Wavelet Transform (LLWT) Stream
To circumvent the irreversible high-frequency loss of max-pooling and strided convolutions, the wavelet stream employs a 2D Learnable Lifting Wavelet Transform. 

#### Polyphase Decomposition (Split Step)
An incoming feature tensor X in R^(C x H x W) is split into four non-overlapping polyphase sub-grids:
- x_00: Even rows, Even columns (Reference component)
- x_01: Even rows, Odd columns (Horizontal component)
- x_10: Odd rows, Even columns (Vertical component)
- x_11: Odd rows, Odd columns (Diagonal component)

#### Stain-Adaptive Spatial Calibration (StainAdapt)
To account for chromatic variance across laboratories, global context descriptor y_gap = GAP(x_00) is recalibrated through a bottleneck MLP with reduction ratio r=4:
    s = sigma( W_2 * ReLU(W_1 * y_gap + b_1) + b_2 )
    S(x_00, f) = f (elementwise-product) s

#### Directional Predict Step
High-frequency detail sub-bands are isolated via depthwise convolutions P_h, P_v, P_d:
    d_h = x_01 - S_h(x_00, P_h(x_00))
    d_v = x_10 - S_v(x_00, P_v(x_00))
    d_d = x_11 - S_d(x_00, P_d(x_00))

#### Directional Update Step
The coarse approximation c_l incorporates detail sub-bands via update depthwise operators U_k and learnable scalars k_h, k_v, k_d:
    c_l = x_00 + k_h * U_h(d_h) + k_v * U_v(d_v) + k_d * U_d(d_d)

#### Exact Invertibility
Because update and predict steps form a triangular uncoupled system, the forward lifting transform is mathematically invertible with zero information loss:
    x_00 = c_l - (k_h * U_h(d_h) + k_v * U_v(d_v) + k_d * U_d(d_d))
    x_01 = d_h + S_h(x_00, P_h(x_00))
    x_10 = d_v + S_v(x_00, P_v(x_00))
    x_11 = d_d + S_d(x_00, P_d(x_00))

In the decoder, an Inverse LLWT module mirrors this formulation using preserved detail sub-bands [d_h, d_v, d_d] routed directly across skip connections.

### 3. Pre-Activation ResNet Stream and GeM Pooling
The semantic stream deploys pre-activation residual blocks:
    x_{l+1} = x_l + F_pre(x_l)
ensuring that gradients backpropagate cleanly via the identity term dL/dx_l = (dL/dx_L) * (I + d/dx_l Sum F_pre).

#### Generalized Mean (GeM) Pooling
Spatial downsampling uses learnable Generalized Mean (GeM) pooling:
    f_GeM(x) = ( (1 / |Omega|) * Sum_{i in Omega} x_i^p )^(1 / p)
where p is a learnable parameter initialized to 3.0. GeM continuously interpolates between average pooling (p -> 1) and max-pooling (p -> infinity), enabling the network to learn feature-specific pooling sharpness.

#### Skip-Connection Feature Alignment Loss (L_feat)
To bridge the semantic disparity between encoder skip features s_l^r and transposed-convolution upsampled decoder features u_l^r, we supervise skip bridges across all four stages l in {1, 2, 3, 4}:
    L_feat = Sum_{l=1}^4 MSE( u_l^r / ||u_l^r||_2, s_l^r / ||s_l^r||_2 )
Because ||u_norm - s_norm||_2^2 = 2 * (1 - cos(theta)), minimizing L_feat maximizes directional alignment in latent space while remaining invariant to arbitrary feature magnitude scaling.

### 4. Multi-Scale Bi-Directional Cross-Attention (BiCA)
Standard spatial self-attention requires an affinity matrix of size (HW) x (HW), entailing (65,536)^2 approx 4.29 x 10^9 elements for 256x256 crops. BiCA resolves this by computing cross-attention in channel space:
- Q = BN(Conv_1x1(x_q)) in R^(C x N)
- K = BN(Conv_1x1(x_kv)) in R^(C x N)
- V = Conv_1x1(x_kv) in R^(C x N)
where N = HW. The channel affinity matrix A in R^(C x C) measures cross-channel covariance:
    A_{i, j} = (1 / sqrt(N)) * < Q_{i, :}, K_{j, :} >
    A_tilde = Softmax_row(A)
    O = A_tilde * V
    x_hat_q = x_q + W_out(O)

#### Zero-Initialized Identity Mapping
By initializing the weight gamma and bias beta of the Batch Normalization layer in W_out to zero, the module functions as an exact identity mapping at step t=0:
    x_hat_q^(0) = x_q^(0) + 0 = x_q^(0)
This prevents early randomized cross-stream updates from corrupting pre-trained ResNet feature representations.

### 5. Sugeno Fuzzy Integral Decision Fusion
Standard linear fusion (averaging, concatenation) assumes spatial homogeneity and additive independence. In histopathology, deep residual features are reliable in smooth nuclear interiors, whereas high-frequency wavelet sub-bands dominate at boundaries.

Let X = {x_1, x_2} denote the wavelet and ResNet streams with predicted probabilities p_1, p_2 in [0, 1]. For singleton fuzzy densities g_1, g_2 in [0, 1], the Sugeno fuzzy integral evaluates to the closed-form expression:
    p_fused = max( min(max(p_1, p_2), g_max), min(p_1, p_2) )
where g_max = g_1 if p_1 >= p_2, else g_2.

The densities g_1 and g_2 are parameterized through unconstrained real-valued logits theta_1, theta_2:
    g_1 = sigma(theta_1),   g_2 = sigma(theta_2)
and optimized end-to-end. The fused probability is converted to logit space via the inverse sigmoid for numerically stable BCE loss computation:
    z_fused = ln(p_fused + epsilon) - ln(1 - p_fused + epsilon)

### 6. Multi-Task Heads and HoVer Distance Regression
To separate clustered nuclei, the network regresses normalized horizontal (h*) and vertical (v*) distance maps to the instance center of mass:
    h*(y, x) = (x - c_x^(k)) / max |x' - c_x^(k)|
    v*(y, x) = (y - c_y^(k)) / max |y' - c_y^(k)|
for (y, x) in Omega_k, and 0 for background.

The HoVer head receives the concatenated final decoder features merged with fine wavelet detail sub-bands:
    F_hv = [d_1 || Interp_2x(d_2) || D_wavelet_fine]
and outputs bounded predictions via tanh activation.

### 7. Composite Loss Formulation
The framework optimizes an end-to-end composite objective:
    L_total = L_seg + lambda_f * L_feat + lambda_hv * L_hv + lambda_b * L_boundary
with lambda_f = 0.1, lambda_hv = 1.0, and lambda_b = 0.15.

- L_seg: Combines Focal Tversky loss (alpha=0.7, beta=0.3, gamma=0.75) to handle severe stroma/nuclear class imbalance with dynamic positive-weighted Binary Cross-Entropy:
    L_seg = L_FT + 0.4 * L_BCE
  Deep supervision applies auxiliary heads at decoder stages 3 and 2:
    L_seg_total = L_seg(z_main) + 0.4 * L_seg(z_aux3) + 0.2 * L_seg(z_aux2)
- L_hv: Combines global MSE with foreground-restricted Mean Squared Gradient Error (MSGE) using Sobel filters:
    L_msge = (1 / |ROI|) * Sum_{ROI} [ (grad_x(T_hat) - grad_x(T*))^2 + (grad_y(T_hat) - grad_y(T*))^2 ]
- L_boundary: Differentiable Sobel gradient consistency loss penalizing L1 discrepancy between predicted and ground-truth edge magnitudes:
    L_boundary = || |grad_S(sigma(z))| - |grad_S(M_gt)| ||_1

## Quantitative Benchmarks

Evaluation is conducted across three clinical histopathology benchmarks: MoNuSeg (Multi-Organ Nuclei Segmentation), TNBC (Triple Negative Breast Cancer), and CPM-15 (Computational Pathology Challenge). Metrics report semantic overlap (Dice) alongside instance metrics (Aggregated Jaccard Index [AJI], Panoptic Quality [PQ]).

| Benchmark Dataset | Method | Dice (%) | AJI (%) | PQ (%) |
|:---|:---|:---:|:---:|:---:|
| MoNuSeg | Swin-UNet | 80.12 | 53.12 | 54.87 |
| MoNuSeg | Trans-UNet | 81.01 | 57.31 | 56.88 |
| MoNuSeg | U2-Net | 80.92 | 56.72 | 56.02 |
| MoNuSeg | HoVer-Net | 82.30 | 61.80 | 60.50 |
| MoNuSeg | DualBranchNet (Ours) | 82.46 | 62.44 | 61.49 |
| TNBC | BoNuS | 76.73 | 60.73 | -- |
| TNBC | Swin-UNet | 84.71 | 61.46 | 63.12 |
| TNBC | Trans-UNet | 86.78 | 63.15 | 65.16 |
| TNBC | HoVer-Net | 86.80 | 65.50 | 64.20 |
| TNBC | DualBranchNet (Ours) | 88.00 | 70.23 | 69.69 |
| CPM-15 | BoNuS | 79.99 | 54.54 | -- |
| CPM-15 | Swin-UNet | 86.92 | 65.79 | 61.28 |
| CPM-15 | Trans-UNet | 88.45 | 68.78 | 63.17 |
| CPM-15 | DualBranchNet (Ours) | 90.02 | 70.82 | 67.40 |

### Modular Ablation Highlights
- Learnable Lifting Wavelet Scheme: Surpasses fixed Haar wavelets (+3.48% Dice, +5.41% AJI on MoNuSeg), confirming that learnable directional filtering preserves sub-pixel nuclear margins.
- Sugeno Fuzzy Fusion: Outperforms feature concatenation, majority voting, and gated attention across all datasets.
- Loss Progression: Adding L_feat steadily improves baseline L_seg, distance regression (L_hv) contributes the primary instance separation gain (+5.0 AJI), and Sobel boundary regularization (L_boundary) maximizes boundary sharpness.

## Installation and Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/mukutchatterjee/dual-branch-wavelet-resnet.git
cd dual-branch-wavelet-resnet

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Core Libraries
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.23.5
- scipy >= 1.10.0
- scikit-image >= 0.20.0
- scikit-learn >= 1.2.0
- albumentations >= 1.3.0
- opencv-python-headless >= 4.7.0
- Pillow >= 9.4.0
- pyyaml >= 6.0

## Dataset Layout

Organize datasets under the `datasets/` root directory using native-resolution images:

```
datasets/
├── MoNuSeg/
│   ├── Train/
│   │   ├── Images/             # Original H&E histology images (.tif / .png)
│   │   └── Masks/              # Instance-level binary or labeled masks
│   └── Test/
│       ├── Images/
│       └── Masks/
├── TNBC/
│   ├── Train/
│   │   ├── Images/
│   │   └── Masks/
│   ├── Validation/
│   │   ├── Images/
│   │   └── Masks/
│   └── Test/
│       ├── Images/
│       └── Masks/
└── CPM15/
    ├── Images/
    └── Masks/
```

Training crops are sampled on-the-fly at native resolution (256x256) via a 4-try foreground-biased sampler. Whole-slide images are never downsampled directly, avoiding resolution degradation.

## Training Protocol

Run end-to-end model training with Automatic Mixed Precision (AMP), gradient clipping (norm 1.0), and Exponential Moving Average (EMA) weight decay:

```bash
python scripts/train.py \
    --config configs/monuseg_config.yaml \
    --device cuda \
    --batch_size 4 \
    --lr 3e-4 \
    --epochs 500 \
    --amp \
    --ema \
    --save_dir checkpoints/monuseg_run
```

Key Training Hyperparameters:
- Optimizer: AdamW (lr 3e-4, weight decay 1e-4, beta1 0.9, beta2 0.999)
- Scheduler: Cosine Annealing with 10-epoch linear warmup
- EMA Decay: Ramp from 0.0 to 0.999 (asymptotic variance reduction factor approx 1/2000)
- Augmentations: Flips, orthogonal rotations, affine shear (+-5 deg), color jitter, Gaussian blur, sensor noise, and random erasing (p=0.2).

## Seamless Inference and Instance Decoding

To evaluate full images without tiling seam artifacts, the inference pipeline applies 2D raised-cosine (Hanning) window stitching across overlapping patches (stride 192, crop size 256):

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/monuseg_run/best_model_ema.pth \
    --dataset MoNuSeg \
    --data_dir datasets/MoNuSeg/Test \
    --patch_size 256 \
    --stride 192 \
    --tune_watershed \
    --output_dir results/monuseg_preds
```

### Instance Separation Pipeline
1. Sliding-Window Blending: Raised-cosine kernel W(i, j) = max(w(i), 0.1) * max(w(j), 0.1) blends overlapping patch predictions.
2. Gradient Energy Surface: Evaluates Sobel gradient magnitudes on horizontal and vertical distance maps:
    G(y, x) = max( |grad(H_hat)|, |grad(V_hat)| )
    G_tilde = (G - min(G)) / (max(G) - min(G) + epsilon)
3. Marker Generation: Identifies seed basins where G_tilde < tau_seed within the semantic mask, followed by morphological disk erosion:
    S_eroded = S_seed (erosion) Disk(r_erode)
4. Watershed Flooding: Floods markers across the energy surface bounded strictly within the fused binary foreground mask.
5. Post-Processing Filter: Spurious instances smaller than 10 pixels are pruned.
6. Validation Tuning: Optimal parameters (tau_seed in {0.3, 0.4, 0.5, 0.6}, r_erode in {0, 1, 2, 3}) are automatically selected via 2D grid search on a 15% validation split.

## Visualizations and Qualitative Outputs

The model generates three synchronized diagnostic representations:

1. Fused Probability Masks: Binary semantic representation produced by Sugeno fuzzy arbitration, maintaining clean margins across varying stain distributions.
2. HoVer Topological Surfaces: Continuous horizontal and vertical distance fields showing clean gradient reversals across abutting cell membranes.
3. Instance Label Matrices: Color-coded nuclear instances resolved by marker-controlled watershed flooding.

## Citation

If you use this codebase or model architecture in your research, please cite the following publication:

```bibtex
@inproceedings{bhattacharjee2024dualbranch,
  title={A Dual-Branch Wavelet-ResNet Model with Cross-Attention and Fuzzy Fusion for Nuclear Segmentation and Instance Segmentation in Histopathology Images},
  author={Bhattacharjee, Diptarko and Chatterjee, Mukut and Kaplun, Dmitrij and Sarkar, Ram},
  booktitle={Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2024},
  organization={IEEE}
}
```

## License and IEEE Terms

This project is licensed under the Apache License 2.0.

In accordance with IEEE conference publication policies and author guidelines, please review the IEEE Intellectual Property Rights and symposium documentation at the [IEEE International Symposium on Biomedical Imaging (ISBI)](https://biomedicalimaging.org/) portal and the [IEEE Open Access Policies](https://www.ieee.org/publications/rights/index.html).

