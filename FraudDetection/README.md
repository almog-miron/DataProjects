# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The project covers the full end-to-end workflow: data-quality checks, exploratory data analysis, model comparison, hyperparameter tuning, cross-validation, feature-importance analysis, decision-threshold optimization, and final evaluation on a held-out test set.

---

## Project Objective

The goal of this project is to build a binary classification model that can identify fraudulent credit card transactions.

The main challenge is the extreme class imbalance: fraudulent transactions represent only about **0.17%** of the dataset.

Because of this imbalance, standard accuracy is not an informative primary metric. A model that predicts every transaction as legitimate would still achieve very high accuracy while detecting no fraud at all.

For that reason, the project focuses mainly on:

- Precision
- Recall
- F1-score
- PR-AUC / Average Precision
- ROC-AUC

PR-AUC is treated as the main model-selection metric because it is especially informative when the positive class is very rare.

---

## Dataset

The dataset contains anonymized European credit card transactions.

The features include:

- `V1`–`V28`: PCA-transformed numerical variables
- `Time`: elapsed time since the first recorded transaction
- `Amount`: transaction amount
- `Class`: target variable
  - `0` = legitimate transaction
  - `1` = fraudulent transaction

The original dataset contained 284,807 transactions.

During data-quality checks, exact duplicate rows were identified and redundant copies were removed before data splitting to reduce the risk of identical observations appearing in both training and evaluation sets.

After cleaning, the dataset contained:

- **283,726 transactions**
- **473 fraudulent transactions**

The fraud rate remained approximately **0.17%**.

> The dataset itself is not included in this repository.  
> Add it locally as `creditcardData.csv` before running the notebook.

---

## Project Workflow

The project follows this general pipeline:

```text
Raw Data
   ↓
Data Quality Checks
   ↓
Duplicate Removal
   ↓
Exploratory Data Analysis
   ↓
Train / Validation / Test Split
   ↓
Baseline Models
   ↓
Tree-Based Models
   ↓
Hyperparameter Optimization
   ↓
Cross-Validation
   ↓
Feature Importance Analysis
   ↓
Model Selection
   ↓
Threshold Optimization
   ↓
Retrain on Train + Validation
   ↓
Final Evaluation on Held-Out Test Set
