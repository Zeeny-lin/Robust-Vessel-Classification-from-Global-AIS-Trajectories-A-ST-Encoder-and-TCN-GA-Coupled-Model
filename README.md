# Robust Vessel Classification from Global AIS Trajectories

This repository contains the main model code for the paper **"Robust Vessel Classification from Global AIS Trajectories: A Spatiotemporal Encoder and TCN-GA Coupled Model"**. The code is reorganized from the latest main multi-head self-attention implementation in `model_v2.py` and split into paper-aligned, reproducible modules.

## Research Objective

The task is global-scale vessel type classification from AIS trajectories for four representative vessel classes:

- Bulk Carrier
- Container Ship
- Fishing
- Oil Tanker

The study addresses three major challenges: unstable global AIS data quality, complex vessel behavior patterns with long-range sequence dependencies, and the need for robust cross-region and cross-ocean generalization.

## Method Overview

The model follows a **Space2Vec + temporal encoding + separate TCN + multi-head self-attention + cross-attention** architecture. This corresponds to the paper's spatiotemporal encoder and TCN-GA coupled framework.

The implementation is organized into six methodological steps:

1. **AIS data cleaning and feature standardization**  
   Latitude, longitude, speed over ground, course over ground, time interval, and daily-cycle features are normalized, interpolated, truncated, and clipped.

2. **Spatiotemporal semantic encoding**  
   Space2Vec-style multi-scale sinusoidal encoding represents geographic locations, while `sog/cog/delta_h/day_frac` are projected into temporal and motion embeddings.

3. **Separate TCN sequence modeling**  
   Spatial and temporal feature branches are processed by independent dilated TCNs to capture long-range dependencies from local maneuvers to cross-day navigation rhythms.

4. **Multi-head self-attention and cross-attention fusion**  
   Spatial and temporal branches are refined by multi-head self-attention, then fused through cross-attention to emphasize behavior-critical trajectory segments.

5. **Ship-level ensemble prediction**  
   The model first predicts trajectory segments, then aggregates segment predictions into vessel-level results using confidence-weighted voting.

6. **Interpretability analysis**  
   Spatial, temporal, cross, and combined attention weights can be exported for identifying the trajectory segments that drive model decisions.

## Repository Structure

```text
vessel-classification-tcn-mha/
  README.md
  requirements.txt
  run_preprocessing.py
  run_experiment.py
  preprocessing/
    preprocess_ais.py      # Standalone six-stage AIS preprocessing pipeline
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

## Reported Results

According to the manuscript, the complete framework achieves the following results on the four-class vessel classification task:

- Overall accuracy: **91.15%**
- Improvement over baseline methods: **5.98% to 27.43%**
- Accuracy with only 30% of the training data: **86.09%**
- Best cross-hemisphere transfer accuracy: **75.81%**
- Best cross-ocean average generalization accuracy: **72.20%**

The ablation study indicates that preprocessing, spatiotemporal encoding, TCN, and GA/attention modules all contribute to final performance.

## Notes

This repository only keeps the main model and reproducible experiment code. Raw AIS data, trained weights, manuscript files, and intermediate experiment outputs are not included. Large data files and `.pth` checkpoints are excluded through `.gitignore` to avoid accidental upload to GitHub.
