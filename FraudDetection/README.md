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
This substantially increased recall but caused precision to collapse, demonstrating that automatic class balancing can create a very large number of false-positive fraud alerts.

Random Forest

Used to capture nonlinear relationships and feature interactions through an ensemble of independently trained decision trees.

XGBoost

Gradient-boosted decision trees were evaluated and optimized using RandomizedSearchCV.

The tuning search included parameters controlling:

tree depth and complexity
learning rate
number of estimators
row and feature subsampling
regularization
minority-class weighting
LightGBM

LightGBM was also evaluated as an alternative gradient-boosting approach.

Its default configuration initially performed poorly, but hyperparameter tuning improved performance substantially, making it highly competitive with XGBoost.

Validation Performance
Model	Precision	Recall	F1	PR-AUC	ROC-AUC
Logistic Regression	0.912	0.684	0.782	0.823	0.968
Logistic Regression - Balanced	0.054	0.895	0.102	0.814	0.975
Random Forest	0.955	0.842	0.895	0.867	0.939
Tuned XGBoost	0.942	0.855	0.897	0.882	0.974
Tuned LightGBM	0.942	0.855	0.897	0.880	0.970

Tuned XGBoost and LightGBM performed very similarly, while Random Forest was also strong. Tuned XGBoost achieved the highest validation PR-AUC.

Cross-Validation

The strongest tree-based models were compared using five-fold stratified cross-validation with Average Precision as the scoring metric.

Model	Mean PR-AUC	Std
Random Forest	0.823	0.040
Tuned XGBoost	0.846	0.024
Tuned LightGBM	0.842	0.030

XGBoost achieved the highest mean cross-validated PR-AUC with relatively low fold-to-fold variability, and was therefore selected as the final candidate model.

Feature Importance

Feature rankings were compared across:

Logistic Regression
Random Forest
XGBoost
LightGBM

Because these algorithms define feature importance differently, raw importance values were not compared directly. Instead, the analysis focused on relative rankings and overlap between the models.

Several variables identified during EDA also appeared among the most important predictors in the fitted models, particularly for Random Forest and XGBoost.

This supported the idea that the univariate differences observed during EDA represented useful predictive signal rather than isolated descriptive effects.

Decision-Threshold Optimization

A classifier's default probability threshold of 0.5 is not necessarily appropriate for fraud detection.

Thresholds were evaluated on the validation set to examine the trade-off between:

catching more fraudulent transactions
limiting false-positive fraud alerts

The final threshold was selected by maximizing F1-score because no explicit business cost function for false positives and false negatives was available.

The selected threshold was approximately:

0.69

In a real fraud-detection system, this threshold should instead be chosen according to the financial and operational costs associated with false positives and missed fraud.

Final Held-Out Test Result

After model and threshold selection were complete, the tuned XGBoost model was retrained using the combined training and validation data and evaluated on the previously unseen test set.

Final performance:

Metric	Score
Precision	0.959
Recall	0.747
F1-score	0.840
PR-AUC	0.826
ROC-AUC	0.975

At the selected operating threshold, the model classified approximately:

71 fraud cases correctly
3 legitimate transactions as fraud
24 fraud cases as legitimate
56,648 legitimate transactions correctly

The model therefore produced extremely few false-positive fraud alerts, although approximately one quarter of fraudulent transactions remained undetected.

This illustrates the central business trade-off in fraud detection: a lower threshold could increase fraud recall, but would also increase the number of legitimate transactions requiring investigation.

Key Takeaways
Extreme class imbalance requires metrics beyond accuracy.
PR-AUC was more useful than ROC-AUC for model selection in this problem.
Automatic class balancing increased recall but caused an unacceptable loss of precision in Logistic Regression.
Hyperparameter tuning had a major impact on boosted-tree performance.
XGBoost and LightGBM performed very similarly after tuning.
Cross-validation helped distinguish between models whose validation scores were very close.
Features highlighted during EDA were also important in several fitted models.
Model probability quality and the final classification threshold are separate decisions.
Threshold selection should ultimately be based on real business costs, not only statistical metrics.
A final untouched test set is important for estimating performance after model-selection decisions have been completed.
Technologies
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
LightGBM
SciPy
Jupyter Notebook
Repository Structure
.
├── fraud_detection.ipynb
├── README.md
└── creditcardData.csv   # not tracked / add locally

The notebook contains the full exploratory analysis, modeling workflow, visualizations, hyperparameter searches, feature analysis, and threshold evaluation.

Running the Project

Clone the repository:

git clone <repository-url>
cd <repository-name>

Install the required packages:

pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm jupyter

Add the dataset to the project directory as:

creditcardData.csv

Then launch Jupyter:

jupyter notebook

and open the fraud-detection notebook.

Author

Almog Miron

Data analysis, machine learning, Python, and statistical modeling.
