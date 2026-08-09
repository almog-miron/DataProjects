# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The project covers the full end-to-end workflow: data-quality checks, exploratory data analysis, model comparison, hyperparameter tuning, cross-validation, feature-importance analysis, decision-threshold optimization, and final evaluation on a held-out test set.

Download CSV at https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models/input

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
''
````markdown
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
````

---

## Exploratory Data Analysis

The exploratory analysis focused on understanding the class imbalance and identifying variables that differ between fraudulent and legitimate transactions.

The analysis included:

* Class-distribution analysis
* Transaction amount comparison
* Temporal transaction patterns
* PCA feature-distribution comparison
* Standardized effect-size analysis using Cohen's d

Because the PCA-transformed variables are anonymized, they cannot be assigned direct business interpretations. Instead, they are evaluated based on statistical separation and predictive relevance.

The PCA components showing the strongest univariate separation between fraudulent and legitimate transactions included:

* `V10`
* `V12`
* `V14`
* `V16`
* `V17`

These features were later compared with model-based feature rankings.

---

## Data Splitting Strategy

The cleaned dataset was split using stratification so that the fraud proportion remained similar across all subsets.

The final split was:

* **64% Training**
* **16% Validation**
* **20% Final Test**

Each subset had a separate purpose:

### Training Set

Used for:

* Model fitting
* Hyperparameter tuning
* Cross-validation

### Validation Set

Used for:

* Model comparison
* Decision-threshold optimization

### Final Test Set

Reserved until all model and threshold decisions were complete.

This design helps reduce data leakage and prevents final model choices from being influenced by test-set performance.

---

## Models Evaluated

Several classification approaches were evaluated.

### Logistic Regression

Logistic Regression was used as a simple linear baseline.

Because Logistic Regression is sensitive to feature scale, the predictors were standardized using `StandardScaler`.

The scaler was fitted only on the training set and then applied to the validation and test sets.

A second Logistic Regression model was trained using:

```python
class_weight="balanced"
```

This greatly increased recall but caused precision to collapse, demonstrating an important point:

> Automatically correcting for class imbalance does not necessarily improve the practical usefulness of a model.

In this case, the balanced model detected more fraud but produced a very large number of false-positive fraud alerts.

---

### Random Forest

Random Forest was evaluated to capture nonlinear relationships and interactions between variables.

Unlike Logistic Regression, tree-based models do not require feature standardization.

Random Forest produced strong performance and provided a useful nonlinear benchmark.

---

### XGBoost

XGBoost was evaluated as a gradient-boosted decision-tree model.

The initial model was followed by hyperparameter optimization using `RandomizedSearchCV`.

The search included parameters controlling:

* Number of estimators
* Learning rate
* Tree depth
* Minimum child weight
* Row subsampling
* Feature subsampling
* Regularization
* Minority-class weighting

The optimization metric was:

```python
scoring="average_precision"
```

so that hyperparameters were selected based on PR-AUC rather than accuracy.

---

### LightGBM

LightGBM was also evaluated as an alternative gradient-boosting algorithm.

Its default configuration initially performed poorly, but hyperparameter tuning substantially improved its performance.

This was an important result: poor default performance did not mean the algorithm was unsuitable for the problem. Instead, the model was highly sensitive to its configuration.

After tuning, LightGBM became highly competitive with XGBoost.

---

## Validation Performance

The main model comparison on the validation set was:

| Model                          | Precision |    Recall |        F1 |    PR-AUC |   ROC-AUC |
| ------------------------------ | --------: | --------: | --------: | --------: | --------: |
| Logistic Regression            |     0.912 |     0.684 |     0.782 |     0.823 |     0.968 |
| Logistic Regression - Balanced |     0.054 |     0.895 |     0.102 |     0.814 |     0.975 |
| Random Forest                  |     0.955 |     0.842 |     0.895 |     0.867 |     0.939 |
| **Tuned XGBoost**              | **0.942** | **0.855** | **0.897** | **0.882** | **0.974** |
| Tuned LightGBM                 |     0.942 |     0.855 |     0.897 |     0.880 |     0.970 |

The three tree-based models performed substantially better than the linear baselines.

Tuned XGBoost and tuned LightGBM were especially close.

XGBoost achieved the highest validation PR-AUC and therefore became the leading candidate.

---

## Cross-Validation

To make sure model selection was not based on a single validation split, the strongest tree-based models were compared using five-fold stratified cross-validation.

Average Precision was used as the cross-validation scoring metric.

The results were:

| Model             | Mean PR-AUC |       Std |
| ----------------- | ----------: | --------: |
| Random Forest     |       0.823 |     0.040 |
| **Tuned XGBoost** |   **0.846** | **0.024** |
| Tuned LightGBM    |       0.842 |     0.030 |

Tuned XGBoost achieved:

* The highest mean cross-validated PR-AUC
* Low fold-to-fold variability
* Strong validation-set performance

LightGBM performed very similarly, while Random Forest had a somewhat lower mean score and greater variability.

Because the difference between XGBoost and LightGBM was small, XGBoost should not be interpreted as dramatically superior. However, it showed the strongest overall combination of validation performance, cross-validation performance, and stability.

XGBoost was therefore selected as the final candidate model.

---

## Feature Importance

Feature importance was compared across:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

Different model families define feature importance differently:

* Logistic Regression uses standardized coefficient magnitude
* Tree-based models use built-in importance measures

Because these values are not directly comparable numerically across model types, the analysis focused on:

* Feature rankings
* Top-ranked features
* Agreement across models

The EDA-selected features were also compared with the top model features.

Several variables identified during exploratory analysis appeared among the most important predictors in the fitted models.

In particular, all five EDA-selected variables appeared among the top 10 features for both Random Forest and XGBoost.

This supports the idea that the differences identified during EDA represented meaningful predictive signal.

---

## Decision-Threshold Optimization

Binary classifiers produce probability scores, which are converted into class predictions using a decision threshold.

The conventional threshold of:

```text
0.50
```

is not necessarily optimal for fraud detection.

Changing the threshold creates a trade-off:

* Lower threshold → higher recall, usually lower precision
* Higher threshold → higher precision, usually lower recall

Threshold optimization was performed exclusively on the validation set.

Several strategies were compared, including:

* Maximizing F1-score
* Maximizing precision while maintaining a minimum recall
* Maximizing recall while maintaining a minimum precision

In the absence of an explicit financial cost function for false positives and false negatives, the threshold maximizing F1-score was selected.

The selected threshold was approximately:

```text
0.69
```

This threshold favored very high precision while maintaining reasonable fraud recall.

In a real production fraud-detection system, the threshold should be selected using actual operational and financial costs rather than statistical metrics alone.

---

## Final Model

After the model and threshold were fully selected, the tuned XGBoost model was retrained using the combined:

* Training set
* Validation set

The final model was then evaluated once on the previously untouched test set.

This ensured that the test set was not involved in model selection or threshold optimization.

---

## Final Held-Out Test Performance

The final XGBoost model achieved:

| Metric    |     Score |
| --------- | --------: |
| Precision | **0.959** |
| Recall    | **0.747** |
| F1-score  | **0.840** |
| PR-AUC    | **0.826** |
| ROC-AUC   | **0.975** |

At the selected threshold, the confusion matrix contained:

|                   | Predicted Legitimate | Predicted Fraud |
| ----------------- | -------------------: | --------------: |
| Actual Legitimate |               56,648 |               3 |
| Actual Fraud      |                   24 |              71 |

This means the model:

* Correctly identified **71 fraudulent transactions**
* Missed **24 fraudulent transactions**
* Incorrectly flagged only **3 legitimate transactions**
* Correctly classified **56,648 legitimate transactions**

The final model therefore achieved extremely high precision while maintaining moderate-to-high recall.

---

## Interpretation of the Final Result

The selected operating threshold produces very few false-positive fraud alerts.

This can be useful in a system where investigating or blocking legitimate transactions is costly.

However, approximately one quarter of fraudulent transactions were still missed.

If the financial cost of missing fraud is much greater than the cost of investigating false positives, a lower threshold could be selected to increase recall.

This demonstrates that fraud detection is not simply a matter of finding the model with the highest score.

The final decision also depends on the operational trade-off between:

* False positives
* False negatives
* Investigation cost
* Fraud loss

---

## Key Findings

The main conclusions from this project are:

1. **Accuracy is inappropriate as the main metric for extremely imbalanced fraud detection.**

2. **PR-AUC is more useful than ROC-AUC for model selection in this dataset.**

3. **Logistic Regression provides a useful baseline but cannot capture the nonlinear structure as effectively as tree-based models.**

4. **Automatic class balancing substantially increased recall but caused precision to collapse.**

5. **Random Forest performed strongly without extensive tuning.**

6. **Hyperparameter optimization had a major effect on gradient-boosting models.**

7. **LightGBM's poor default performance was primarily configuration-related.**

8. **XGBoost and LightGBM produced very similar final model-selection performance.**

9. **Cross-validation helped distinguish between models whose validation scores were almost identical.**

10. **Several features identified during EDA also appeared among the most important model predictors.**

11. **Choosing a classification threshold is a separate problem from training the classifier itself.**

12. **The optimal production threshold should depend on real business costs rather than F1-score alone.**

13. **Keeping a final untouched test set is essential for estimating performance after model-selection decisions are complete.**

---

## Technologies Used

### Data Analysis

* Python
* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM

### Statistical Analysis

* SciPy

### Environment

* Jupyter Notebook

---

## Repository Structure

```text
credit-card-fraud-detection/
│
├── credit_card_fraud_detection.ipynb
├── README.md
├── requirements.txt
└── creditcardData.csv        # local only / not tracked by Git
```

The notebook contains the complete analysis, including:

* Data cleaning
* EDA
* Visualizations
* Model training
* Hyperparameter searches
* Cross-validation
* Feature-importance analysis
* Threshold optimization
* Final test evaluation

---

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd credit-card-fraud-detection
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm jupyter
```

Alternatively, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Place the dataset in the project directory using the filename:

```text
creditcardData.csv
```

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
credit_card_fraud_detection.ipynb
```

and run the notebook from top to bottom.

---

## Notes

The PCA-transformed variables in this dataset are anonymized, so their original business meanings are unavailable.

As a result, feature-importance analysis in this project is focused on predictive relevance rather than direct business interpretation.

The final threshold was selected statistically because no real fraud-investigation cost function was available. In a production environment, threshold selection should be based on actual fraud losses and operational investigation costs.

---

## Author

**Almog Miron**

Data Analysis · Machine Learning · Statistical Modeling · Python

```
```

