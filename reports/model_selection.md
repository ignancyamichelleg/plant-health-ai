# Model selection

Ranked on the validation split by macro-F1, ties broken by the lowest
per-class recall. The test split is not used for selection.

| Rank | Model | Macro-F1 | Accuracy | Min class recall | Recall ALS | Recall rust | Recall healthy |
|---|---|---|---|---|---|---|---|
| 1 | baseline_cnn | 0.849 | 0.850 | 0.778 | 0.841 | 0.778 | 0.932 |
| 2 | cnn_aug | 0.824 | 0.827 | 0.659 | 0.659 | 0.889 | 0.932 |
| 3 | mobilenet_frozen | 0.808 | 0.812 | 0.636 | 0.636 | 0.844 | 0.955 |

**Selected: `baseline_cnn`**
