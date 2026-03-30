# Smart Energy Forecasting & Anomaly Detection — Project Results

## Dataset

- **Source**: UCI Appliances Energy Prediction Dataset
- **Total Samples**: 19,687 rows × 60 columns
- **Engineered Features**: 57
- **Target Variable**: `Appliances` (energy consumption in Wh)
- **Train/Val/Test Split**: 70% / 15% / 15% (13,780 / 2,953 / 2,954 rows)
- **Validation Strategy**: Chronological split — never shuffle time series

---

## Part A — EDA & Feature Engineering

### Key Findings

- **12 EDA visualizations** produced with business/engineering interpretations
- **8 feature engineering groups** created:
  - Time features (hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos)
  - Fourier features
  - Lag features (lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48, lag_144)
  - Rolling statistics (rolling_mean_6, rolling_std_6, rolling_mean_24, rolling_std_24, rolling_mean_144, rolling_std_144)
  - Temperature/Humidity differentials
  - Interaction features
- **Pearson vs Mutual Information** correlation analysis completed
- Engineered dataset saved as `data/df_engineered.csv`

---

## Part B — Classical Machine Learning Results

### From-Scratch Models (NumPy Only)

| Model | MAE (Wh) | RMSE (Wh) | MAPE (%) | R² |
|---|---|---|---|---|
| Normal Equation | 29.6911 | 61.2902 | 28.32% | 0.5459 |
| Batch GD (LR=0.01) | 27.9159 | 61.9755 | 24.90% | 0.5357 |

- **Normal Equation training time**: 53.12 ms
- **Batch Gradient Descent training time**: 128.85 ms
- **Top predictor (Normal Eq)**: `lag_1` with weight 74.63

### Sklearn Models with Optuna Tuning

| Model | MAE (Wh) | RMSE (Wh) | MAPE (%) | R² |
|---|---|---|---|---|
| Lasso Regression (α=0.1) | 28.6180 | 60.8488 | 26.75% | 0.5525 |
| Lasso (Optuna α=0.5736) | 26.9667 | 60.7622 | 24.09% | 0.5537 |
| Random Forest (Optuna) | 34.6473 | 62.7995 | 33.76% | 0.5233 |
| LightGBM (Optuna) | 34.7460 | 62.3322 | 36.22% | 0.5304 |

### Best Classical ML Model

**Lasso (Optuna-tuned, α=0.5736)** achieves the best overall performance:
- Lowest MAE: 26.97 Wh
- Lowest RMSE: 60.76 Wh
- Lowest MAPE: 24.09%
- Highest R²: 0.5537

---

## Part C — Deep Learning Results

### Architecture Summary

| Model | Architecture | Parameters |
|---|---|---|
| LSTM-64 | 2-layer LSTM (64 hidden), MSE loss | — |
| LSTM-128 (MSE) | 2-layer LSTM (128 hidden), MSE loss | — |
| LSTM-128 (Huber) | 2-layer LSTM (128 hidden), Huber loss | — |
| CNN-LSTM (k=3) | Conv1D(k=3) + 2-layer LSTM(128) | — |
| CNN-LSTM (k=7) | Conv1D(k=7) + 2-layer LSTM(128) | — |

### DL Model Comparison Table

| Model | MAE (Wh) | RMSE (Wh) | MAPE (%) | R² |
|---|---|---|---|---|
| LSTM-128 (Huber) | 39.1100 | 81.2158 | 35.68% | 0.2004 |
| LSTM-64 | 37.3550 | 81.2468 | 31.96% | 0.1998 |
| CNN-LSTM (k=7) | 35.4119 | 81.7597 | 27.58% | 0.1896 |
| LSTM-128 (MSE) | 45.5641 | 83.1618 | 46.34% | 0.1616 |
| CNN-LSTM (k=3) | 38.3903 | 86.8896 | 29.85% | 0.0848 |

### Best DL Model

**CNN-LSTM (k=7)** achieves the lowest MAPE (27.58%) and MAE (35.41 Wh), suggesting that larger kernel sizes capture more temporal context.

### Key DL Observations

- **Huber loss** outperforms MSE loss, indicating energy consumption data has outliers that MSE penalizes too heavily
- **CNN-LSTM architectures** capture local temporal patterns that pure LSTMs miss
- **DL models generally underperform classical ML** on this dataset (R² ~0.08–0.20 vs ~0.52–0.55), likely due to dataset size limitations

---

## Part D — Anomaly Detection Results

### Anomaly Injection Summary

| Type | Count | Method |
|---|---|---|
| Point | 20 | Random extreme values (mean ± 5σ) |
| Contextual | 29 | High consumption during 3–5 AM |
| Collective | 24 | 2 sustained blocks of 12 steps each |
| **Total** | **73 / 2,954 (2.47%)** | |

### Overall Anomaly Detection Comparison

| Model | Precision | Recall | F1-Score | PR-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.0060 | 0.0411 | 0.0104 | 0.0200 |
| **One-Class SVM** | **0.1961** | **0.9726** | **0.3264** | **0.2299** |
| LSTM Autoencoder | 0.0247 | 0.4110 | 0.0467 | 0.0210 |

### Per-Anomaly-Type F1 Comparison

| Anomaly Type | Isolation Forest F1 | One-Class SVM F1 | LSTM Autoencoder F1 |
|---|---|---|---|
| Point | 0.0077 | 0.1094 | 0.0149 |
| Contextual | 0.0038 | 0.1662 | 0.0147 |
| Collective | 0.0000 | 0.1416 | 0.0197 |

### Best Anomaly Detection Model

**One-Class SVM** significantly outperforms both Isolation Forest and LSTM Autoencoder:
- Highest Recall: 97.26% — catches almost all anomalies
- Highest F1-Score: 0.3264
- Highest PR-AUC: 0.2299
- Best F1 across all anomaly types

### Key Anomaly Detection Observations

- **PR-AUC** is used instead of ROC-AUC because anomalies are rare (~2.5%), making precision-recall a harder and more meaningful benchmark
- **One-Class SVM** excels at high recall but has lower precision (19.61%), meaning it flags many normal points as anomalies
- **LSTM Autoencoder** shows moderate recall (41.10%) but very low precision (2.47%)
- **Isolation Forest** struggles with both precision and recall on this dataset

---

## Project Structure

```
ML&DL_Assignment/
├── data/                           # Processed datasets
│   ├── energydata_complete.csv     # Raw UCI dataset
│   ├── df_engineered.csv           # Feature-engineered dataset
│   ├── X_train_seq.npy             # DL training sequences
│   ├── X_val_seq.npy               # DL validation sequences
│   ├── X_test_seq.npy              # DL test sequences
│   ├── y_train_seq.npy             # DL training targets
│   ├── y_val_seq.npy               # DL validation targets
│   └── y_test_seq.npy              # DL test targets
├── models/                         # Trained model files
│   ├── best_lstm64.pt              # Best LSTM-64 model
│   ├── best_lstm128_mse.pt         # Best LSTM-128 (MSE) model
│   ├── best_lstm128_huber.pt       # Best LSTM-128 (Huber) model
│   ├── best_cnn_lstm_k3.pt         # Best CNN-LSTM (k=3) model
│   ├── best_cnn_lstm_k7.pt         # Best CNN-LSTM (k=7) model
│   ├── best_autoencoder.pt         # Best LSTM Autoencoder model
│   ├── best_lgbm_model.pkl         # Best LightGBM model
│   ├── scaler_X.pkl                # Feature scaler (classical ML)
│   ├── dl_scaler_X.pkl             # Feature scaler (DL)
│   └── dl_scaler_y.pkl             # Target scaler (DL)
├── images/                         # Generated visualizations
│   ├── weights_normal_eq.png       # Normal Equation feature weights
│   ├── gd_convergence.png          # GD learning rate convergence
│   ├── actual_vs_pred_scratch.png  # From-scratch model predictions
│   ├── optuna_lasso_param_importance.png
│   ├── optuna_rf_history.png
│   ├── optuna_rf_param_importance.png
│   ├── optuna_lgbm_history.png
│   ├── optuna_lgbm_param_importance.png
│   ├── actual_vs_predicted_all.png # All model predictions
│   ├── residual_analysis.png       # Residual analysis
│   ├── feature_importance_comparison.png
│   ├── shap_beeswarm.png           # SHAP beeswarm plot
│   ├── shap_summary_bar.png        # SHAP feature importance
│   ├── shap_dependence_plots.png
│   ├── pearson_vs_mi.png           # Correlation analysis
│   ├── rank_agreement_scatter.png
│   ├── sample_window.png           # DL sequence window example
│   ├── all_loss_curves.png         # DL training loss curves
│   ├── loss_fn_comparison.png      # MSE vs Huber comparison
│   ├── per_step_error.png          # DL per-step error analysis
│   ├── dl_dashboard.png            # DL evaluation dashboard
│   ├── anomaly_injection_plot.png  # Anomaly injection visualization
│   ├── anomaly_comparison.png      # Anomaly detection comparison
│   ├── ae_loss_curve.png           # Autoencoder loss curve
│   ├── ae_reconstruction_error.png # Autoencoder reconstruction error
│   └── detection_timeline.png      # Detection timeline
├── notebook_A_EDA_Features.py      # Part A: EDA & Feature Engineering
├── notebook_A_EDA_Features.ipynb
├── notebook_B_Classical_ML.py      # Part B: Classical ML
├── notebook_B_Classical_ML.ipynb
├── notebook_C_DeepLearning.py      # Part C: Deep Learning
├── notebook_C_DeepLearning.ipynb
├── notebook_D_AnomalyDetection.py  # Part D: Anomaly Detection
├── notebook_D_AnomalyDetection.ipynb
├── objective_file/                 # Project requirements
└── results.md                      # This file
```

---

## Summary & Conclusions

1. **Classical ML outperforms Deep Learning** on this dataset for energy consumption forecasting. The Optuna-tuned Lasso model (R²=0.5537) significantly beats the best DL model CNN-LSTM k=7 (R²=0.1896). This is likely because the dataset (~20K samples) is too small for DL to generalize effectively.

2. **Lag features dominate** all models — `lag_1` (consumption 10 min ago) is consistently the most important predictor across Normal Equation, Lasso, Random Forest, LightGBM, and SHAP analysis.

3. **Huber loss improves DL robustness** — LSTM-128 with Huber loss achieves higher R² (0.2004) than with MSE loss (0.1616), confirming that robust loss functions are better suited for energy data with outliers.

4. **One-Class SVM is the strongest anomaly detector** — with 97.26% recall, it catches nearly all injected anomalies, though at the cost of many false positives (precision=19.61%).

5. **Chronological validation** is critical for time series — using `TimeSeriesSplit` and strict chronological train/val/test splits prevents data leakage that would inflate metrics.
