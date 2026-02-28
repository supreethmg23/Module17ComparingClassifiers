# Module17ComparingClassifiers

# Comparing Classifiers: Bank Term Deposit Subscription Prediction

## Overview

This project focuses on predicting whether a customer will subscribe to a bank term deposit using a bank marketing dataset. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, feature selection, and evaluation of multiple classification models. A major challenge addressed in this study is the significant imbalance in the target variable.

---

## Data Preprocessing

### Dataset

The dataset initially contained **41,188 rows × 21 columns**, including features such as age, job, marital status, education, and campaign details. After removing 12 duplicate rows, the final cleaned dataset comprised **41,176 rows**.

### Cleaning Steps

- **Duplicates:** Removed duplicate entries.
- **Handling 'unknown' values:**  
  Categorical columns containing `'unknown'` were imputed using their respective mode values.  
  - Example:  
    - `job` → replaced with `"admin."`  
    - `marital` → replaced with `"married"`
- **Data Quality Checks:**  
  Verified the `age` column to ensure values fall within a realistic range `[0, 100]`.

---

## Exploratory Data Analysis (EDA)

### Target Variable Distribution

The target variable (`y`) indicates whether a client subscribed to a term deposit.

The dataset is **highly imbalanced**, with a majority labeled `"no"` and a minority labeled `"yes"`.

---

### Categorical Variable Analysis

Count plots were generated for:

- `job`
- `marital`
- `education`
- `contact`
- `poutcome`

Key observations:

- **Job:** "admin.", "blue-collar" dominate.
- **Marital:** "married" is the most common category.
- **Contact:** "cellular" is the primary communication channel.

Insight: First-contact effectiveness is crucial since many customers are new to bank marketing efforts.

---

### Continuous Variable Analysis

Histograms and boxplots were analyzed for:

- `age`
- `duration`
- `campaign`
- `pdays`

Key findings:

- **Call Duration:** Longer durations correlate with higher subscription rates.
- **Euribor3m:** Lower values are associated with higher subscription likelihood.
- Many clients have `pdays = 999`, indicating no previous contact.
- Most clients had very few campaign contacts.

---

## Feature Engineering & Preprocessing

### Engineered Features

The following features were created:

- `target_demographic` – Flag for key job categories (e.g., "admin.", "blue-collar", "technician") for married clients.
- `is_cellular` – Indicates if the contact method is cellular.
- `prev_success` – Flag for previously successful campaigns.
- `middle_age` – Flag for customers aged between 30 and 50.
- `long_duration` – Flag for call durations longer than 200 seconds.
- `new_contact` – Flag indicating new contacts (`pdays = 999`).
- `high_education` – Flag for customers with higher education credentials.
- `job_marital` – Interaction feature combining job and marital status.

---

### Data Transformation

- Target variable `y` encoded as:
  - `"yes"` → 1  
  - `"no"` → 0
- Categorical features were one-hot encoded.
- Numerical features were standardized using `StandardScaler`.

---

### Train-Test Split

The dataset was split into:

- 80% Training Set
- 20% Testing Set

---

## Feature Selection

Two feature selection methods were applied:

### 1. SelectKBest (Mutual Information)

Top features:

- `age`
- `duration`
- `pdays`
- `previous`
- `emp.var.rate`
- `cons.price.idx`
- `cons.conf.idx`
- `euribor3m`
- `nr.employed`
- `is_cellular`

---

### 2. Random Forest Feature Importances

Most important features:

- `duration`
- `euribor3m`
- `age`
- `nr.employed`
- `long_duration`

---

## Modeling and Evaluation

### Classifiers Evaluated

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest

---

### Performance Metrics

Models were evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- ROC-AUC Score

---

### Example Results

**Logistic Regression**
- Accuracy ≈ 90.3%
- ROC-AUC ≈ 0.9155
- Lower recall and F1-score for minority class

**Random Forest**
- ROC-AUC ≈ 0.9403 (Best performance)
- Strong discrimination capability

---

## Handling Class Imbalance

To address imbalance:

- Applied `RandomUnderSampler` on training data.

Result:

- Improved precision, recall, and F1-score for minority class.
- Trade-off between overall accuracy and minority detection performance.

---

## Conclusions

### Key Insights

- Longer call durations increase subscription likelihood.
- Lower Euribor3m rates are associated with higher subscriptions.
- Target imbalance significantly impacts model performance.
- Domain-based feature engineering improved predictive power.

---

### Modeling Outcome

While most classifiers achieved high overall accuracy, **Random Forest** delivered the highest ROC-AUC score. However, imbalance remains a key limitation.

---

## Future Work

- Apply resampling techniques such as SMOTE or oversampling.
- Use cost-sensitive learning.
- Perform hyperparameter tuning for Random Forest and other models.
- Explore ensemble techniques tailored for imbalanced datasets.