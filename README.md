# Robust Vessel Classification from Global AIS Trajectories: A Spatiotemporal Encoder and TCN-GA Coupled Model

**Author:** 

Jialiang Gao, ADC (Fujian), Fuzhou University, Fuzhou, 350108, China  
Zhenyi Lin, ADC (Fujian), Fuzhou University, Fuzhou, 350108, China  

##
This repository contains the main model code for the paper **"Robust Vessel Classification from Global AIS Trajectories: A Spatiotemporal Encoder and TCN-GA Coupled Model"**. 

## Research Objective

The task is global-scale vessel type classification from AIS trajectories for four representative vessel classes:

- Bulk Carrier
- Container Ship
- Fishing
- Oil Tanker
  
**Appen Fig. C.1.Trajectory Heatmap.**

![Trajectory Heatmap](https://github.com/Zeeny-lin/Robust-Vessel-Classification-from-Global-AIS-Trajectories-A-ST-Encoder-and-TCN-GA-Coupled-Model/blob/main/docs/figures/Appen%20Fig%20C.1.png)

The study addresses three major challenges: unstable global AIS data quality, complex vessel behavior patterns with long-range sequence dependencies, and the need for robust cross-region and cross-ocean generalization.

## Method Overview

The model follows a **ST encoding + TCN-GA** architecture. The implementation is organized into six methodological steps:

1. **AIS data cleaning and feature standardization**  
   Latitude, longitude, speed over ground, course over ground, time interval, and daily-cycle features are normalized, interpolated, truncated, and clipped.

2. **Spatiotemporal semantic encoding**  
   Space2Vec-style multi-scale sinusoidal encoding represents geographic locations, while `sog/cog/delta_h/day_frac` are projected into temporal and motion embeddings.

3. **TCN sequence modeling**  
   Independent dilated TCNs process spatial and temporal feature branches to capture long-range dependencies from local maneuvers to cross-day navigation rhythms.

4. **Global-Attention fusion**  
   Spatial and temporal branches are refined by multi-head self-attention, then fused through cross-attention to emphasize behavior-critical trajectory segments.

5. **Ship-level ensemble prediction**  
   The model first predicts trajectory segments, then aggregates segment predictions into vessel-level results using confidence-weighted voting.

6. **Interpretability analysis**  
   Attention weights can be exported for identifying the trajectory segments that drive model decisions.

## Model Framework

**Fig. 1. Overall TCN-GA vessel classification framework.**

![Overall Space2Vec-TCN-GA vessel classification framework](https://github.com/Zeeny-lin/Robust-Vessel-Classification-from-Global-AIS-Trajectories-A-ST-Encoder-and-TCN-GA-Coupled-Model/blob/main/docs/figures/Fig%201.png)

**Fig. 2. Six-stage AIS trajectory preprocessing workflow.**

![Six-stage AIS trajectory preprocessing workflow](https://github.com/Zeeny-lin/Robust-Vessel-Classification-from-Global-AIS-Trajectories-A-ST-Encoder-and-TCN-GA-Coupled-Model/blob/main/docs/figures/Fig%202.png)

**Fig. 3. TCN-Global Attention module.**

![TCN-Global Attention module](https://github.com/Zeeny-lin/Robust-Vessel-Classification-from-Global-AIS-Trajectories-A-ST-Encoder-and-TCN-GA-Coupled-Model/blob/main/docs/figures/Fig%203.png)

## Repository Structure

```text
vessel-classification-tcn-mha/
  README.md
  requirements.txt
  run_preprocessing.py
  run_experiment.py
  preprocessing/
    preprocess_ais.py      # Standalone six-stage AIS preprocessing pipeline
  comparative_experiments/
    README.md              # Baseline and ablation source map
    external_baselines/    # Compared models grouped by experimental paradigm
    internal_ablations/    # Component-level ablation scripts
  src/
    config.py              # Experiment parameters, class names, and paths
    data_pipeline.py       # Data loading, cleaning, standardization, and padding
    model.py               # Space2Vec-TCN-MHA main model
    train.py               # Training loop, early stopping, and Top-K checkpoints
    evaluate.py            # Segment prediction and ship-level voting evaluation
    attention_analysis.py  # Attention weight export and visualization
  docs/
    method_steps.md        # Paper-style method step description
```

## Data Format

Raw AIS data can be placed in class-specific folders. The preprocessing script preserves this folder structure:

```text
raw_ais/
  Bulk Carrier/*.csv
  Container Ship/*.csv
  Fishing/*.csv
  Oil Tanker/*.csv
```

After preprocessing, the final processed data root is:

```text
data/preprocessed/06_compressed/
```

For training, the expected split layout is:

```text
data/data/process_seg/
  train/
    Bulk Carrier/*.csv
    Container Ship/*.csv
    Fishing/*.csv
    Oil Tanker/*.csv
  val/
  test/
```

Each CSV file should contain, or allow the code to derive, the following fields:

```text
lat, lon, sog, cog, delta_h, day_frac
```

If a `postime` column is available, `delta_h` and `day_frac` are computed automatically.

## AIS Preprocessing

The paper uses a six-stage AIS preprocessing pipeline:

1. Time-threshold trajectory segmentation
2. Statistical cleaning with speed and sliding-window filters
3. PCHIP interpolation with circular COG handling
4. Fixed-length trajectory splitting
5. Kalman smoothing
6. Speed and distance based compression (SBC)

Run preprocessing with:

```bash
python run_preprocessing.py \
  --raw-root raw_ais \
  --output-root data/preprocessed \
  --time-gap-hours 24 \
  --speed-threshold-kn 30 \
  --interpolation-seconds 4800 \
  --max-segment-points 300 \
  --kalman-accuracy-m 1.5 \
  --sbc-distance-m 1000 \
  --sbc-speed-kn 7
```

The final processed root can then be split into `train/val/test` folders and passed to `run_experiment.py`.

## Quick Start

```bash
pip install -r requirements.txt
python run_experiment.py --data-root data/data/process_seg --output-dir runs/tcn_mha
```

Optional arguments:

```bash
python run_experiment.py \
  --data-root data/data/process_seg \
  --output-dir runs/tcn_mha \
  --epochs 50 \
  --batch-size 16 \
  --max-seq-len 300
```

## Results

According to the manuscript, the complete framework achieves the best performance on the four-class vessel classification task.

### Overall Model Comparison

| Model Type | Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Feature Engineering | LightGBM | 84.63 | 84.82 | 84.63 | 84.70 |
| Feature Engineering | Random Forest | 85.71 | 86.54 | 85.83 | 85.70 |
| Feature Engineering | SVM | 82.91 | 83.45 | 82.86 | 82.99 |
| Feature Engineering | BP-AdaBoost | 85.71 | 85.92 | 85.72 | 85.74 |
| Image Encoding | ResNet | 73.48 | 72.70 | 73.17 | 72.70 |
| Image Encoding | MVFFNet | 72.57 | 74.98 | 72.44 | 72.38 |
| Image Encoding | DCN | 71.53 | 74.34 | 71.43 | 71.44 |
| Image Encoding | EfficientNet | 73.30 | 71.89 | 72.56 | 71.89 |
| Sequence Modeling | STGNN | 83.99 | 84.22 | 84.24 | 84.02 |
| Sequence Modeling | TrAISformer | 86.01 | 86.66 | 86.27 | 86.46 |
| Sequence Modeling | TimeMachine | 78.26 | 79.27 | 78.89 | 79.08 |
| Proposed | TCN-GA | **91.15** | **91.34** | **91.16** | **91.20** |

### Preprocessing Ablation

| Data Setting | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Delta Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Data | 78.57 | 80.51 | 78.57 | 78.88 | - |
| + Segmentation | 86.22 | 86.63 | 86.22 | 86.61 | +7.65 |
| + Cleaning | 86.48 | 88.75 | 86.48 | 86.81 | +0.26 |
| + Interpolation | 87.50 | 90.11 | 87.50 | 87.96 | +1.02 |
| + Splitting | 87.60 | 88.55 | 87.90 | 88.22 | +0.10 |
| + Smoothing | 88.01 | 89.96 | 88.01 | 88.33 | +0.41 |
| + Compression | 88.62 | 88.85 | 88.62 | 88.67 | +0.61 |

### Component Ablation

| Model Variant | Without Encoders | Without Spatial Encoder | Without Temporal Encoder | Full Spatiotemporal Encoder |
| --- | ---: | ---: | ---: | ---: |
| Channel Attention | 69.90 | 82.40 | 78.06 | 82.91 |
| Cross Attention | 69.60 | 82.95 | 83.54 | 84.97 |
| Global Attention | 78.66 | 85.20 | 84.95 | 86.73 |
| TCN | 82.54 | 83.62 | 85.98 | 89.87 |
| TCN + Channel Attention | 82.14 | 87.55 | 88.60 | 90.13 |
| TCN + Cross Attention | 83.44 | 88.36 | 88.86 | 89.88 |
| TCN + Global Attention | 84.55 | 88.62 | 88.90 | **91.15** |

### Robustness to Training Data Sparsity

| Training Ratio | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| 100% | 91.15 | 91.34 | 91.08 | 91.20 |
| 75% | 90.12 | 90.72 | 90.04 | 90.26 |
| 60% | 87.59 | 88.23 | 87.51 | 87.73 |
| 30% | 86.09 | 86.60 | 86.01 | 86.24 |
| 20% | 78.10 | 79.57 | 78.02 | 78.46 |
| 10% | 78.36 | 79.73 | 78.28 | 78.80 |

### Spatial Generalization

| Scenario | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Eastern Hemisphere -> Western Hemisphere | 75.81 | 79.32 | 75.82 | 76.62 |
| Western Hemisphere -> Eastern Hemisphere | 62.64 | 64.83 | 62.66 | 61.33 |
| Cross-hemisphere average | 69.23 | - | - | - |
| Best cross-ocean average | 72.20 | - | - | - |

The ablation study indicates that preprocessing, spatiotemporal encoding, TCN, and GA/attention modules all contribute to final performance.

### Interpretability Analysis
**Fig. 5. The global attention weights reflect discriminative behaviour characteristics of trajectories corresponding vessel type.**

![The global attention weights reflect discriminative behaviour characteristics of trajectories corresponding vessel type](https://github.com/Zeeny-lin/Robust-Vessel-Classification-from-Global-AIS-Trajectories-A-ST-Encoder-and-TCN-GA-Coupled-Model/blob/main/docs/figures/Fig%205.png)

