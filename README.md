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
## 📊 Training Results
### Classification Report (Training Data)
```markdown
              precision    recall  f1-score   support
           0       0.71      0.71      0.71    604084
           1       0.51      0.50      0.50    288708
           2       0.69      0.68      0.69    423067

    accuracy                           0.66   1315859
   macro avg       0.63      0.63      0.63   1315859
weighted avg       0.66      0.66      0.66   1315859
```

### Confusion Matrix (Training Data)
```lua
[[431134  99401  73549]
 [ 89041 144874  54793]
 [ 91049  42554 289464]]
```

---

## 📊 Test Results (Cost-Sensitive Evaluation)
After applying **cost-sensitive learning**, the model was evaluated on **test data**.

### Classification Report (Test Data)
```markdown
              precision    recall  f1-score   support
           0       0.68      0.73      0.70   1630942
           1       0.47      0.46      0.46    868897
           2       0.73      0.67      0.70   1422856

    accuracy                           0.65   3922695
   macro avg       0.62      0.62      0.62   3922695
weighted avg       0.65      0.65      0.65   3922695
```

### Confusion Matrix (Test Data)
```lua
[[1187031  279058  164853]
 [ 276578  398375  193944]
 [ 290559  173536  958761]]
```

### Additional Test Metrics
- **Macro-F1 Score:** `0.62`
- **Macro Precision:** `0.62`
- **Macro Recall:** `0.62`

---

## 🚀 How to Use This Model

### 1️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

### 2️⃣ Load the model in Python:
```python
import joblib
model = joblib.load("rf_optimized_optuna.joblib")
```

### 3️⃣ Predict cybersecurity incidents:
```python
sample_input = [...]  # Replace with actual input features
prediction = model.predict(sample_input)
print("Predicted Class:", prediction)
```

---

## 🔮 Future Improvements
✅ **Feature Engineering:** Introduce new features for better classification.  
✅ **Hyperparameter Optimization:** Further fine-tune model parameters.  
✅ **Deep Learning Integration:** Explore neural networks for complex patterns.  
✅ **Real-time Deployment:** Implement the model in a production environment.  

---

## 👥 Contributors
| Name | Role |
|------|------|
| 🧑‍💻 [Khatalahmed] | Data Scientist & ML Engineer |


