# Telecom Churn Prediction — Capstone Part 2 & Part 3

## Project Overview

This project builds a complete machine learning pipeline to predict customer churn-related outcomes using supervised learning, ensemble models, hyperparameter tuning, cross-validation, and model serialization.

The objective is to compare different machine learning models, identify the most robust model, tune its parameters, and package the final model into a reusable sklearn Pipeline.

---

# Dataset

Dataset used:

`cleaned_data.csv`

Data preprocessing steps performed:

* Removed duplicate records
* Handled missing values using median imputation
* Converted data types appropriately
* Performed feature engineering
* Applied encoding for categorical variables
* Split data into training and testing sets

---

# Part 2 — Supervised Machine Learning

## Models Implemented

### Logistic Regression

A Logistic Regression model was trained as the baseline classification model.

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

# Part 3 — Advanced Modeling

## 1. Decision Tree Baseline

A default Decision Tree classifier was trained using:

```python
DecisionTreeClassifier(max_depth=None)
```

The model achieved high training accuracy but lower testing accuracy, showing signs of overfitting.

Decision trees are high-variance models because they make greedy splits based on training data and do not revisit earlier decisions. A deep tree can memorize noise in the training dataset.

---

# 2. Controlled Decision Tree

A regularized decision tree was trained using:

```python
max_depth=5
min_samples_split=20
```

### Hyperparameter Explanation

**max_depth**

Controls how deep the tree can grow. Lower depth reduces variance and overfitting but may increase bias.

**min_samples_split**

Controls the minimum number of samples required before splitting a node. It prevents the tree from creating splits based on very small noisy groups.

The controlled tree showed a smaller train-test accuracy gap compared with the unrestricted decision tree.

---

# 3. Gini vs Entropy Comparison

Two decision trees were trained using:

* Gini impurity
* Entropy

## Gini Impurity Formula

[
Gini = 1-\sum p_i^2
]

A Gini value of 0 means all samples in a node belong to the same class.

## Entropy Formula

[
Entropy=-\sum p_i log_2(p_i)
]

Entropy measures the uncertainty or randomness inside a node.

---

# 4. Random Forest Model

Random Forest was trained with:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

Evaluation metrics:

* Training accuracy
* Testing accuracy
* ROC-AUC

## Feature Importance

The model feature_importances_ attribute was used to identify the top five important features.

Random Forest feature importance is calculated by measuring the average reduction in Gini impurity produced by each feature across all trees.

Unlike linear regression coefficients, feature importance values do not represent direction or magnitude of relationship. They only indicate how useful a feature is for splitting the data.

---

# Bagging Concept

Random Forest uses bagging (Bootstrap Aggregation).

Each decision tree receives a random bootstrap sample of the training data with replacement. During each split, only a random subset of features is considered.

By combining predictions from many different trees, Random Forest reduces variance and improves generalization compared with a single deep decision tree.

---

# 5. Gradient Boosting

Gradient Boosting Classifier was trained using:

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

The model was evaluated using:

* Training accuracy
* Testing accuracy
* ROC-AUC

The Gradient Boosting model was also included in cross-validation comparison.

---

# 6. Feature Ablation Study

The five lowest importance features from Random Forest were removed.

A second Random Forest model was trained using the reduced feature set.

Comparison:

| Model                         | Test ROC-AUC |
| ----------------------------- | ------------ |
| Full Random Forest            | Add value    |
| Reduced Feature Random Forest | Add value    |

## Interpretation

If the reduced model achieves similar or higher AUC, the removed features were likely uninformative and added noise.

If AUC decreases, the removed features contributed useful information.

A simpler model can reduce:

* Inference cost
* Storage requirements
* Maintenance complexity

However, feature removal is acceptable only if performance degradation remains below the business tolerance level.

---

# 7. Cross Validation Comparison

Models evaluated using:

```python
StratifiedKFold(
n_splits=5,
shuffle=True,
random_state=42
)
```

Metric:

`ROC-AUC`

Results:

| Model                    | Mean AUC  | Standard Deviation |
| ------------------------ | --------- | ------------------ |
| Logistic Regression      | Add value | Add value          |
| Controlled Decision Tree | Add value | Add value          |
| Random Forest            | Add value | Add value          |
| Gradient Boosting        | Add value | Add value          |

Cross-validation provides a more reliable estimate of model performance because the model is evaluated on multiple different validation splits instead of depending on a single train-test split.

---

# 8. Hyperparameter Tuning — GridSearchCV

Random Forest pipeline:

```python
SimpleImputer
StandardScaler
RandomForestClassifier
```

Parameter grid:

```python
{
'n_estimators':[50,100,200],
'max_depth':[5,10,None],
'min_samples_leaf':[1,5]
}
```

Total configurations:

```
3 × 3 × 2 = 18 models

18 × 5 folds = 90 training runs
```

Grid Search performs exhaustive testing of all combinations, while Randomized Search tests only a selected number of combinations and is faster for large parameter spaces.

---

# 9. Learning Curve Analysis

The best GridSearchCV pipeline was trained using:

* 20%
* 40%
* 60%
* 80%
* 100%

training data fractions.

Results:

| Training Fraction | Training AUC | Test AUC  |
| ----------------- | ------------ | --------- |
| 20%               | Add value    | Add value |
| 40%               | Add value    | Add value |
| 60%               | Add value    | Add value |
| 80%               | Add value    | Add value |
| 100%              | Add value    | Add value |

## Conclusion

Training AUC trend shows model variance behaviour.

If test AUC continues increasing at 100%, the model is data-limited.

If test AUC reaches a plateau, the model is mainly limited by model capacity.

---

# Model Serialization

The final tuned pipeline was saved using Joblib.

Save:

```python
joblib.dump(best_pipeline,"best_model.pkl")
```

Load and predict:

```python
loaded_model = joblib.load("best_model.pkl")

predictions = loaded_model.predict(sample_rows)

print(predictions)
```

The saved model successfully loaded and generated predictions.

Example output:

```
[0 1]
```

---

# Final Model Recommendation

| Model               | CV Mean AUC | CV Std AUC | Test AUC  |
| ------------------- | ----------- | ---------- | --------- |
| Logistic Regression | Add value   | Add value  | Add value |
| Decision Tree       | Add value   | Add value  | Add value |
| Random Forest       | Add value   | Add value  | Add value |
| Gradient Boosting   | Add value   | Add value  | Add value |

The final model recommendation is the tuned Random Forest pipeline because it provided strong predictive performance while reducing overfitting compared with a single decision tree. Cross-validation results were used to confirm that the model generalizes well across different data splits. The pipeline structure also makes deployment easier because preprocessing and prediction steps are stored together. The final trained pipeline was serialized as `best_model.pkl` for future use.

---

# Repository Structure

```
telecom-churn-capstone-part2-part3/

│
├── capstone_2_and_3.ipynb
├── cleaned_data.csv
├── best_model.pkl
└── README.md
```
