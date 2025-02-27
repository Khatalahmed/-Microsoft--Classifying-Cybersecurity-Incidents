# Machine Learning Model Comparison and Optimization

This repository contains the code and results for comparing multiple machine learning models, optimizing the best-performing model, and evaluating its performance on both training and test datasets. The goal was to identify the best model based on accuracy and Macro-F1 score, perform hyperparameter tuning, and analyze the results.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Model Comparison](#model-comparison)
3. [Best Model Selection](#best-model-selection)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Train Data Results](#train-data-results)
6. [Test Data Results](#test-data-results)
7. [Cost-Sensitive Evaluation](#cost-sensitive-evaluation)
8. [Usage](#usage)
9. [Dependencies](#dependencies)
10. [License](#license)

---

## 🚀 Project Overview
This project involves:
- Comparing multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Gradient Boosting) on a classification task.
- Selecting the best model based on **Macro-F1 Score** and **Accuracy**.
- Performing hyperparameter tuning using Optuna.
- Evaluating the optimized model on both training and test datasets.
- Saving the trained model for future use.

---

## 📊 Model Comparison
The following table summarizes the performance of the models:

| Model               | Accuracy | Macro-F1 Score | Precision | Recall |
|---------------------|----------|----------------|-----------|--------|
| Logistic Regression | 0.5393   | 0.50           | 0.50      | 0.51   |
| Decision Tree       | 0.6609   | 0.64           | 0.64      | 0.65   |
| Random Forest       | 0.6636   | 0.64           | 0.64      | 0.65   |
| XGBoost             | 0.6745   | 0.61           | 0.69      | 0.60   |
| LightGBM            | 0.6415   | 0.62           | 0.62      | 0.62   |
| Gradient Boosting   | 0.6399   | 0.53           | 0.70      | 0.55   |

---

## 🏆 Best Model Selection
The **Random Forest** model was selected as the best model based on its **Macro-F1 Score** and **Accuracy**.

### Best Model Performance:
- **Accuracy**: 0.6636
- **Macro-F1 Score**: 0.64
- **Precision**: 0.64
- **Recall**: 0.65

### Best Hyperparameters:
```python
{
    'n_estimators': 240,
    'max_depth': 28,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'bootstrap': False,
    'max_features': 'sqrt'
}
