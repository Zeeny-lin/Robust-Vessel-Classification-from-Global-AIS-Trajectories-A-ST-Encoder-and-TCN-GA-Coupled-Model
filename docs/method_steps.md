# Method Steps

This folder reorganizes the latest multi-head self-attention model from `model_v2.py` into paper-style modules.

1. Data quality control and feature standardization
   - Run the six-stage preprocessing script in `preprocessing/preprocess_ais.py`.
   - Segment raw AIS trajectories by large timestamp gaps.
   - Remove physically implausible speed jumps and local coordinate outliers.
   - Resample uneven AIS messages with PCHIP interpolation.
   - Split long trajectories, smooth coordinates with Kalman filtering, and compress redundant points with SBC.

2. Spatiotemporal semantic encoding
   - Encode latitude and longitude with Space2Vec-style multi-scale sinusoidal features.
   - Project motion and temporal features (`sog`, `cog`, `delta_h`, `day_frac`) into a temporal embedding.

3. Long-sequence dependency modeling
   - Apply separate dilated TCN branches to spatial and temporal embeddings.
   - Preserve long-range AIS behavioral dependencies without recurrent training instability.

4. Multi-head attention fusion
   - Use multi-head self-attention independently on spatial and temporal TCN outputs.
   - Use cross-attention to fuse spatial queries with temporal keys and values.
   - Keep attention weights for interpretability.

5. Classification and ship-level voting
   - Predict segment-level vessel classes.
   - Aggregate segment predictions into ship-level results by confidence-weighted voting.

6. Interpretability analysis
   - Export spatial, temporal, cross, and combined attention weights.
   - Plot attention against behavioral variables such as speed.
