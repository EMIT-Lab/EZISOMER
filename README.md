# EZISOMER

**Fluorescence-image analysis and RGB-based E/Z concentration prediction**

EZISOMER provides a workflow for 96-well fluorescence-image localization, paired RGB-based concentration regression, and ANN feature interpretation. The regression scripts use **one CSV dataset** throughout. There are no dataset-combination modes, concentration-range exclusions, or automatically defined experiment subsets.

## Files

| File | Description |
| --- | --- |
| `CNN-detect-circle.py` | CNN well localization (original component; unchanged). |
| `data.csv` | **Single input dataset** supplied by the user. |
| `rgb_common.py` | Shared data loading, composition calculation, metrics, and plotting. |
| `train_traditional_ml.py` | Ten traditional regression models; five-fold CV by default. |
| `train_ann.py` | Multi-output ANN; leave-one-group-out CV by default, with optional random five-fold CV. |
| `ann_feature_importance_permutation.py` | Permutation importance for the saved ANN. |
| `README.md` / `README_CN.md` | Documentation. |

The three regression/interpretability scripts consume tabular RGB measurements, not photographs directly. The CNN component runs separately.

## Installation

Python 3.10+ is recommended.

```bash
pip install numpy pandas scikit-learn matplotlib torch joblib
```

The CNN stage additionally uses `torchvision`, `opencv-python`, `albumentations`, and `tqdm`, depending on its source environment. GPU acceleration is optional for the ANN.

## Stage I — CNN well localization

*Original description retained.* `CNN-detect-circle.py` implements a **KeypointNet** for locating 96 well centers. The architecture consists of four convolutional blocks followed by three fully connected layers predicting 192 x/y values. The image-processing pipeline applies rotations, flips, and brightness augmentation. Outputs include model weights (`best_model.pth`), training history, validation results, and predicted keypoint visualizations.

Historical input conventions: `.jpg` images in `DATA_96/circle/` and coordinate `.npz` files in `DATA_96/loc/`.

```bash
python CNN-detect-circle.py
```

## Single dataset: `data.csv`

All regression and importance analyses use the **same file**, with the following required columns:

| Column | Meaning |
| --- | --- |
| `A_B`, `A_G`, `A_R` | Blue, green, red measurements for input A. |
| `B_B`, `B_G`, `B_R` | Blue, green, red measurements for input B. |
| `Y1` | Reference E concentration. |
| `Y2` | Reference Z concentration. |
| `图片名称` | Group identifier used by the ANN's default leave-one-group-out CV. |

`排组`, `列号`, `A_L`, `B_L`, or additional metadata may remain in the CSV; they do not enter the model. For random five-fold CV, `图片名称` is not required.

Example:

```csv
图片名称,A_B,A_G,A_R,B_B,B_G,B_R,Y1,Y2
sample_01,180,72,18,142,128,35,4,1
sample_01,182,75,20,144,125,36,1,4
sample_02,184,80,21,147,122,38,2,2
```

All valid rows are retained as provided. The scripts require numeric, finite features and nonnegative concentration labels; preprocessing is limited to train-fold scaling where applicable. Use a consistent concentration unit within `data.csv`.

### Composition metric

The ANN and traditional regressors predict **E and Z concentrations**, then calculate:

\[
 de = \frac{C_E-C_Z}{C_E+C_Z}.
\]

For predicted composition, negative model-estimated concentrations are clipped to zero **only for the ratio calculation**; reported E/Z concentration predictions remain unmodified. When the ratio denominator is zero, `de` is mathematically undefined (`NaN`) and is omitted from *de-only metrics*, while the record remains in concentration training and concentration metrics. No composition-specific threshold or label is imposed. Consequently, these `de` scores are not numerically identical to earlier postprocessed results.

## Stage II — Traditional machine learning

**Script:** `train_traditional_ml.py`

Implemented algorithms: KNN, Random Forest, Extra Trees, Decision Tree, Gradient Boosting, RBF-SVR, Linear Regression, Ridge, Lasso, and PLS. Each estimator predicts `Y1` and `Y2`; `de` is calculated from these two predictions. Scaling is fit separately on each training fold, including target scaling.

Default: randomized **five-fold K-fold CV**, `random_state=42`.

```bash
python train_traditional_ml.py --data data.csv --pdf
```

Run selected models:

```bash
python train_traditional_ml.py --data data.csv --models KNN RandomForest SVR_RBF
```

Optional grouped evaluation on the **same dataset**:

```bash
python train_traditional_ml.py --data data.csv --cv loao --group-column 图片名称
```

Output directory: `results_traditional_kfold/` (or `results_traditional_loao/`). It contains `SUMMARY_Train_vs_Test_R2.csv`, `test_OOF_predictions_all_models.csv`, `fold_training_predictions_all_models.csv`, `fold_train_test_metrics.csv`, saved `.joblib` estimators, and PNG/PDF graphs.

## Stage III — ANN concentration prediction

**Script:** `train_ann.py`

- Six RGB inputs → hidden layers **32 → 16 → 8** → two concentration outputs.
- ReLU activations, Dropout `0.10` after each of the first two hidden layers.
- Adam optimizer, up to 4,000 epochs, batch size 8, patience 200 for early stopping.
- Feature and target scaling are fitted from training records only.

Default evaluation is **leave-one-group-out CV** (LOAO): the `图片名称` value is used as the group key. Each distinct key is held out once. Epoch selection uses a separate validation split inside each training fold; the held-out test group is not used for early stopping.

```bash
python train_ann.py --data data.csv --cv loao --pdf
```

To compare the **same dataset** under random five-fold CV:

```bash
python train_ann.py --data data.csv --cv kfold --folds 5 --pdf
```

Quick execution test (not for scientific reporting):

```bash
python train_ann.py --data data.csv --quick --skip-plots
```

ANN outputs: `SUMMARY_Train_vs_Test_R2.csv`, `test_OOF_predictions.csv`, `fold_training_predictions.csv`, `fold_train_test_metrics.csv`, `fold_details.csv`, `training_history.csv`, the final `final_ann_model.pth`, and `final_scaler_X.pkl` / `final_scaler_y.pkl`. LOAO also writes `LOAO_metrics_BY_GROUP.csv`.

## Stage IV — ANN permutation feature importance

**Script:** `ann_feature_importance_permutation.py`

Load the final trained ANN; independently permute each of the six RGB features across the records in `data.csv` while keeping ANN parameters unchanged. Importance is the **increase in MAE** over the unchanged inputs, measured over repeated permutations (50 by default), and reported as mean ± SD. Separate results are produced for E, Z, total concentration, and calculated `de`.

```bash
python ann_feature_importance_permutation.py --data data.csv --results results_ann_loao --repeats 50 --pdf
```

Outputs: `permutation_repeats.csv`, `permutation_importance_summary.csv`, and plots. The analysis uses the final model on the provided dataset; it represents **in-sample model dependence**, not independent test-fold importance. Normalized importance percentages are descriptive and not physical contributions.

## Metrics and interpretation

Each training script reports a single overall dataset label, **`all_data`**, with:

- `Train_R2_fold_mean`: mean training R² across CV folds.
- `Test_R2_OOF_pooled`: pooled out-of-fold test R²; one held-out prediction per input record.
- `Test_MAE_OOF`, `Test_RMSE_OOF`, `Test_Bias_OOF`.

Four prediction targets are reported: `E_concentration`, `Z_concentration`, `total_concentration`, and `de_from_concentration`. The target and feature definitions are the same in both training scripts. Constant-valued target groups do not have a meaningful R².

Model comparison is best performed with the **same CSV and the same `--cv` option**. Different CV schemes answer different validation questions and need not yield identical scores.

## Example directory

```text
EZISOMER/
├── CNN-detect-circle.py
├── data.csv
├── rgb_common.py
├── train_traditional_ml.py
├── train_ann.py
├── ann_feature_importance_permutation.py
├── README.md
└── README_CN.md
```

## Research use

The code does not impose chemical class-specific row selection. Any scientific inclusion/exclusion policy should be applied transparently when preparing the input CSV, before running these scripts. Model hyperparameters are listed in source code and should be documented when reporting results.
