# Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The project covers the full modeling workflow: data-quality checks, exploratory data analysis, model comparison, hyperparameter tuning, cross-validation, feature-importance analysis, decision-threshold optimization, and final evaluation on a held-out test set.

## Project Objective

The goal is to identify fraudulent credit card transactions while dealing with an extreme class imbalance: fraud represents only approximately **0.17%** of the observations.

This makes standard accuracy a poor evaluation metric. A model predicting every transaction as legitimate would still achieve extremely high accuracy while detecting no fraud.

For that reason, the analysis focuses primarily on:

- Precision
- Recall
- F1-score
- PR-AUC / Average Precision
- ROC-AUC

PR-AUC is treated as the primary model-selection metric because it is more informative for a problem with a very rare positive class.

---

## Dataset

The dataset contains anonymized European credit card transactions.

Features include:

- `V1`–`V28`: PCA-transformed numerical features
- `Time`: elapsed time since the first recorded transaction
- `Amount`: transaction amount
- `Class`: binary target
  - `0` = legitimate transaction
  - `1` = fraudulent transaction

Exact duplicate copies were removed before data splitting to reduce the risk of identical observations appearing in both training and evaluation sets.

After cleaning, the dataset contained **283,726 transactions**, including **473 fraud cases**.

> The dataset itself is not included in this repository.  
> Add the dataset locally as `creditcardData.csv` before running the notebook.

---

## Project Workflow

### 1. Data Quality & Exploratory Analysis

The initial analysis included:

- missing-value checks
- duplicate detection
- class-distribution analysis
- transaction amount comparison
- temporal fraud patterns
- PCA feature-distribution analysis
- standardized effect-size comparison using Cohen's d

The PCA components showing the strongest univariate separation between fraudulent and legitimate transactions included:

`V10`, `V12`, `V14`, `V16`, and `V17`.

Because the PCA features are anonymized transformations, they are interpreted in terms of statistical and predictive relevance rather than business meaning.

---

### 2. Train / Validation / Test Strategy

The cleaned dataset was split using stratification to preserve the fraud proportion:

- **64% Training**
- **16% Validation**
- **20% Final Test**

The subsets were used for different purposes:

- **Training:** model fitting, cross-validation, and hyperparameter optimization
- **Validation:** model comparison and decision-threshold selection
- **Test:** final evaluation only after all modeling decisions were finalized

This separation was used to reduce leakage and avoid selecting models or thresholds based on final test performance.

---

## Models Evaluated

The following classification approaches were compared:

### Logistic Regression

Used as a simple linear baseline.

A second Logistic Regression model was tested using:

```python
class_weight="balanced"
