# Loan Eligibility Prediction

End-to-end machine learning project that predicts loan approval status (`Y` or `N`) from applicant demographic, financial, and credit history data using `scikit-learn` pipelines.

## Overview
This project builds and compares two classification models—**Support Vector Classifier (SVC)** and **Random Forest Classifier**—to predict whether a loan application will be approved (`Y`) or rejected (`N`).

All preprocessing and modeling steps are wrapped in a `ColumnTransformer` and `Pipeline` architecture so transformations are learned strictly on training data, preventing data leakage and keeping inference reproducible.

## Dataset
- **Source:** Standard Loan Prediction dataset (Analytics Vidhya / Kaggle style)
- **Target:** `Loan_Status` (`Y` = Approved, `N` = Rejected)
- **Features:** `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, and `Property_Area` (`Loan_ID` is dropped from model features and retained only for submission output)

## Approach
1. **Exploratory Data Analysis:** Inspect target class balance (`Loan_Status`) and income distributions across approval outcomes (`ApplicantIncome`).
2. **Leak-Free Preprocessing (`ColumnTransformer` + `Pipeline`):**
   - **Numerical features:** Median imputation (`SimpleImputer(strategy="median")`) + feature scaling (`StandardScaler`).
   - **Categorical features:** Most-frequent imputation (`SimpleImputer(strategy="most_frequent")`) + one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`).
3. **Validation & Model Training:** Evaluate models on a stratified validation split (and/or cross-validation) using:
   - **Support Vector Classifier (`SVC`):** RBF kernel with `class_weight="balanced"` and `probability=True`.
   - **Random Forest Classifier:** 200 trees (`n_estimators=200`), `max_depth=10`, and `class_weight="balanced"`.
4. **Evaluation & Submission Export:** Compare precision, recall, F1-score, and accuracy on validation data, then generate test set prediction CSVs (`loan_svc_submissions.csv` and `loan_rf_submissions.csv`).

## Model Comparison: Random Forest vs. SVC
- **Random Forest:** Captures non-linear feature interactions and skewed income/loan thresholds naturally without relying on distance geometry, and is less sensitive to unscaled or skewed continuous features. Depth limiting (`max_depth=10`) helps control overfitting on tabular data.
- **SVC:** Finds maximum-margin decision boundaries in scaled feature space and works well on clean, moderately sized datasets, but is more sensitive to feature scaling, outliers, and high dimensionality created by one-hot encoding.

## Results (Validation / Cross-Validation)
| Model | Validation / CV Accuracy | Notes |
| :--- | :--- | :--- |
| **Random Forest** | *Add validation/CV score* | Strong non-linear splits; easy to inspect feature importance |
| **SVC** | *Add validation/CV score* | Relies on `StandardScaler` + one-hot features; balanced class weights |

*(Replace placeholders with your exact validation/cross-validation metrics from your run.)*

## Project Structure
```
loan-eligibility-prediction/
├── data/
│   ├── train_data.csv
│   └── test_data.csv
├── src/
│   └── train_models.py
├── .gitignore
├── README.md
└── requirements.txt