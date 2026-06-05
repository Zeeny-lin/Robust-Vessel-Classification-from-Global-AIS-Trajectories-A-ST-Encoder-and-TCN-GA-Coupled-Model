# Comparative Experiments

This folder contains the code used for comparisons against external baseline models and for internal ablation studies. It is separated from the main `src/` implementation so that the GitHub repository keeps the proposed Space2Vec-TCN-MHA model clean while still preserving the experimental evidence reported in the manuscript.

The scripts were copied from the latest experiment versions found in the workspace. Some baseline scripts keep their original local path assumptions, so update dataset paths before running them on a new machine.

## Folder Layout

```text
comparative_experiments/
  external_baselines/
    feature_engineering/
    image_encoding/
    sequence_modeling/
  internal_ablations/
```

## External Baselines

The external baselines follow the three comparison groups used in the paper: feature-engineering models, image-encoding deep models, and sequence-modeling deep models.

| Group | Baseline | Local Code | Compared Paper or Repository |
| --- | --- | --- | --- |
| Feature engineering | BP-AdaBoost | `external_baselines/feature_engineering/bp_adaboost.py` | Han et al. (2025), as cited in the manuscript baseline table; the local script implements the BP-AdaBoost comparison setting used in the experiments. |
| Feature engineering | Random Forest | `external_baselines/feature_engineering/random_forest.py` | Huang et al. (2023), "Ship classification based on AIS data and machine learning methods." |
| Feature engineering | SVM | `external_baselines/feature_engineering/svm.py` | Yan et al. (2022), "Ship classification and anomaly detection based on spaceborne AIS data considering behavior characteristics." |
| Feature engineering | Extra Trees | `external_baselines/feature_engineering/extra_trees.py` | Additional classical tree-ensemble baseline implemented locally; this is not one of the main manuscript table models. |
| Image encoding | MVFFNet | `external_baselines/image_encoding/mvffnet.py` | Liang, Zhan, and Liu (2021), "MVFFNet: Multi-view feature fusion network for imbalanced ship classification." |
| Image encoding | DCN | `external_baselines/image_encoding/dcn.py` | Guo and Xie (2022), "Research on ship trajectory classification based on a deep convolutional neural network." |
| Sequence modeling | STGNN | `external_baselines/sequence_modeling/stgnn.py` | Feng et al. (2022), "IS-STGCNN: An Improved Social spatial-temporal graph convolutional neural network for ship trajectory prediction." |
| Sequence modeling | TrAISformer | `external_baselines/sequence_modeling/traisformer/` | Nguyen and Fablet (2021), "TrAISformer: A Transformer Network with Sparse Augmented Data Representation and Cross Entropy Loss for AIS-based Vessel Trajectory Prediction." |
| Sequence modeling | TimeMachine | `external_baselines/sequence_modeling/timemachine_supervised/` | Ahamed and Cheng (2024), "TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting." The original README and license copied with the code are preserved as `timemachine_README_original.md` and `timemachine_LICENSE`. |

The manuscript also discusses LightGBM, ResNet, and EfficientNet as comparison models. In the current workspace, these models were not found as independent final Python scripts, so they are not uploaded in this code-only release. The included files are the standalone comparison implementations that were available as reusable code.

## Internal Ablations

The internal ablation scripts test which parts of the proposed method contribute to the final result.

| Ablation | Local Code | Removed or Isolated Component |
| --- | --- | --- |
| Attention only | `internal_ablations/attention_only.py` | Keeps the attention branch while removing the full proposed spatiotemporal stack. |
| No TCN Transformer | `internal_ablations/no_tcn_transformer.py` | Removes the TCN module from the proposed sequence model. |
| Space2Vec only | `internal_ablations/space2vec_only.py` | Keeps only the spatial encoding pathway. |
| Temporal only | `internal_ablations/temporal_only.py` | Keeps only the temporal encoding pathway. |
| Space2Vec + GA without TCN | `internal_ablations/space2vec_ga_no_tcn.py` | Tests the spatial encoder and global-attention setting without TCN. |
| Temporal + GA without Space2Vec | `internal_ablations/temporal_ga_no_space2vec.py` | Tests the temporal encoder and global-attention setting without Space2Vec. |
| Spatiotemporal + GA without TCN | `internal_ablations/spatiotemporal_ga_no_tcn.py` | Keeps both encoders and global attention while removing TCN. |
| TCN + global attention without encoders | `internal_ablations/tcn_global_attention_no_encoders.py` | Tests the TCN and attention backbone without the spatial and temporal encoders. |

## Reference List

- Meyer, R., and Kleynhans, W. (2024). Vessel classification using AIS data. *Ocean Engineering*, 319, 120043. https://doi.org/10.1016/j.oceaneng.2024.120043
- Huang, I., Lee, M., Nieh, C., and Huang, J. (2023). Ship classification based on AIS data and machine learning methods. *Electronics*, 13(1), 98. https://doi.org/10.3390/electronics13010098
- Yan, Z., Song, X., Zhong, H., Yang, L., and Wang, Y. (2022). Ship classification and anomaly detection based on spaceborne AIS data considering behavior characteristics. *Sensors*, 22(20), 7713. https://doi.org/10.3390/s22207713
- Liang, M., Zhan, Y., and Liu, R. W. (2021). MVFFNet: Multi-view feature fusion network for imbalanced ship classification. *Pattern Recognition Letters*, 151, 26-32. https://doi.org/10.1016/j.patrec.2021.07.024
- Guo, T., and Xie, L. (2022). Research on ship trajectory classification based on a deep convolutional neural network. *Journal of Marine Science and Engineering*, 10(5), 568. https://doi.org/10.3390/jmse10050568
- Feng, H., Cao, G., Xu, H., and Ge, S. S. (2022). IS-STGCNN: An Improved Social spatial-temporal graph convolutional neural network for ship trajectory prediction. *Ocean Engineering*, 266, 112960. https://doi.org/10.1016/j.oceaneng.2022.112960
- Nguyen, D., and Fablet, R. (2021). TrAISformer: A Transformer Network with Sparse Augmented Data Representation and Cross Entropy Loss for AIS-based Vessel Trajectory Prediction. *arXiv*. https://doi.org/10.48550/arxiv.2109.03958
- Ahamed, M. A., and Cheng, Q. (2024). TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting. *arXiv*. https://doi.org/10.48550/arxiv.2403.09898
