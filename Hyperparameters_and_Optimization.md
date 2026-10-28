
## Training Setup and Implementation Details

The model is trained using an ImageNet-pretrained ResNet backbone, while all additional modules are initialized using Kaiming initialization. Training is performed on foreground-biased image crops to increase the representation of relevant structures and improve segmentation learning. Exponential Moving Average (EMA) and gradient clipping are employed to promote stable optimization.

### 1. Model Initialization

| Component | Configuration |
|---|---|
| Backbone | ResNet |
| Backbone weights | ImageNet-pretrained |
| Additional modules | Kaiming initialization |
| Input crop size | 256 × 256 pixels |
| Crop sampling strategy | Foreground-biased |

The pretrained backbone provides transferable visual representations, while Kaiming initialization is used for newly introduced modules that do not inherit pretrained weights.

### 2. Training Configuration

| Parameter | Value |
|---|---:|
| Training duration | 500 epochs |
| Batch size | 4 |
| Optimizer | AdamW |
| Initial learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻³ |
| Gradient clipping | Enabled |
| Maximum gradient norm | 1.0 |
| Exponential Moving Average (EMA) | Enabled |
| EMA decay | 0.999 |

AdamW is used for parameter optimization with a learning rate of 3 × 10⁻⁴ and a weight decay of 10⁻³. Gradient clipping limits the gradient norm to 1.0, helping prevent excessively large parameter updates. EMA tracks a smoothed version of the model parameters using a decay factor of 0.999.

### 3. Validation Split

A validation split comprising 15% of the available data is used to select the watershed post-processing configuration. The Aggregated Jaccard Index (AJI) serves as the selection criterion, with the configuration achieving the highest validation AJI retained for evaluation.

| Setting | Configuration |
|---|---|
| Validation fraction | 15% |
| Selection metric | Aggregated Jaccard Index (AJI) |
| Tuning target | Watershed post-processing |
| Selection rule | Highest validation AJI |

### 4. Watershed Post-processing

Watershed post-processing is tuned to improve the separation of touching instances and produce more accurate instance-level segmentation masks. Two parameters are evaluated: the seed threshold and the erosion radius. Small predicted instances with an area below 10 pixels are discarded.

| Parameter | Symbol | Candidate Values |
|---|---|---|
| Seed threshold | τ_seed | {0.3, 0.4, 0.5, 0.6} |
| Erosion radius | r_erode | {0, 1, 2, 3} |
| Minimum instance area | — | 10 pixels |
| Small-instance filtering | — | Discard instances with area < 10 pixels |
| Optimization metric | — | Validation AJI |

The search evaluates all 16 combinations of seed threshold and erosion radius. The final configuration is selected based on validation AJI rather than the test-set performance.

### 5. Reproducibility Summary

| Category | Setting |
|---|---|
| Backbone initialization | ImageNet-pretrained ResNet |
| Additional module initialization | Kaiming |
| Training crop size | 256 × 256 pixels |
| Crop sampling | Foreground-biased |
| Epochs | 500 |
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 10⁻³ |
| Gradient clipping norm | 1.0 |
| EMA decay | 0.999 |
| Validation split | 15% |
| Watershed search combinations | 16 |
| Post-processing selection metric | AJI |

### Implementation Notes

- **Validation-based tuning:** Watershed parameters are selected using the validation split to avoid tuning directly on the test set.
- **Stable optimization:** Gradient clipping and EMA are used to support stable training and reduce sensitivity to individual parameter updates.
- **Instance-level evaluation:** AJI is used to select post-processing parameters because it accounts for the overlap between predicted and ground-truth instances.
- **Transparent reporting:** The reported configuration describes the specified experimental setup. Additional implementation details should be documented where applicable to support exact reproduction.
