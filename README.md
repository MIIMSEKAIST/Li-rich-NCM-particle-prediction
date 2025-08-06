# Li-Rich Primary‐Particle Size Prediction  
A complete, reproducible pipeline for predicting the **mean primary particle size** of Li-rich layered cathode materials from composition and synthesis parameters, complete with probabilistic uncertainty estimates and SHAP-based interpretation.

<table>
  <tr>
    <td><strong>Built With</strong></td>
    <td>Python 3.9 | scikit-learn | NGBoost | MatImputer | SHAP | seaborn/matplotlib</td>
  </tr>
  <tr>
    <td><strong>Key Outputs</strong></td>
    <td>.csv datasets, trained <code>.pkl</code> models &amp; scalers, parity / calibration / uncertainty plots, SHAP visualisations</td>
  </tr>
</table>

---

## 1. Project Overview
1. **Imputation** – `imputer.py` fills missing values in nine core features using four strategies  
   (MatImputer, Mean, K-NN, MICE).   
2. **Model Training & Evaluation** – `ml_model.py` (invoked by `main.py`) trains NGBoost regressors  
   with 10-fold CV, saves the best model & scaler, and produces extensive diagnostic plots.   
3. **Uncertainty-Aware Interpretation** – `ngboost_shap_interpretation.py` generates feature-importance  
   bars, SHAP summary/dependence/interaction plots for both **loc** (mean) and **scale** (uncertainty)  
   outputs.   
4. **Utility Plots** – `utils.py` houses all plotting helpers with a central `PlotConfig` dataclass.   
5. **EDA** – `EDA.ipynb` (optional) offers quick visual sanity checks before modelling.

---

## 2. Repository Structure
```text
├── dataset/
│   ├── Li-rich data collection_full.csv
│   ├── Li-rich data collection_half.csv
│   └── (generated) Li-rich_train_* & Li-rich_test_*.csv
├── models/
│   └── models_<dataset_type>/        # saved .pkl models & scalers
├── plots/                            # auto-generated figures
├── imputer.py
├── main.py                           # CLI entrypoint
├── ml_model.py
├── ngboost_shap_interpretation.py
├── utils.py
└── EDA.ipynb
