# Logistic Regression on Titanic Dataset

## Overview
This project demonstrates the cleaning, preprocessing, and training of a Logistic Regression model on a Titanic dataset. The objective is to predict whether a passenger survived the Titanic disaster based on specific features.

---

## Dataset Description
The dataset contains the following features:
- **survived**: Target variable indicating survival (0 = No, 1 = Yes).
- **pclass**: Passenger class (1 = First, 2 = Second, 3 = Third).
- **sex**: Gender of the passenger.
- **age**: Age of the passenger.
- **sibsp**: Number of siblings/spouses aboard.
- **parch**: Number of parents/children aboard.
- **fare**: Ticket fare.
- **embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## Steps in the Workflow

### 1. **Data Loading**
- Loaded the dataset using Pandas from Google Drive.

### 2. **Data Cleaning**
- **Dropped unnecessary columns**:
  - `class`, `who`, `adult_male`, `deck`, `embark_town`, `alive`, `alone`, `pclass_redundant`, `fare_age_combination`.
- **Handled missing values**:
  - Filled missing `age` values with the median.
  - Filled missing `fare` values with the median.
  - Filled missing `embarked` values with the mode.
- **Filtered age outliers**:
  - Retained rows where `age` ≤ 100.

### 3. **Feature Encoding**
- Normalized the `sex` column to lowercase.
- Encoded `sex` and `embarked` columns using `LabelEncoder`.

### 4. **Feature Selection**
- Selected the following columns for modeling:
  - Features (`X`): `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`.
  - Target (`y`): `survived`.

### 5. **Model Training**
- Split the dataset into training and test sets using an 80-20 split and stratified sampling to handle class imbalance.
- Trained a Logistic Regression model with the following parameters:
  - `random_state=42` for reproducibility.
  - `max_iter=1000` to ensure convergence.

### 6. **Evaluation**
- Predicted survival on the test set.
- Evaluated the model using:
  - **Accuracy**: 80%.
  - **Confusion Matrix**:
    ```
    [[3 1]
     [2 9]]
    ```
  - **Classification Report**:
    ```
                  precision    recall  f1-score   support

               0       0.60      0.75      0.67         4
               1       0.90      0.82      0.86        11

        accuracy                           0.80        15
       macro avg       0.75      0.78      0.76        15
    weighted avg       0.82      0.80      0.81        15
    ```

---

## Key Findings
- The Logistic Regression model achieved an accuracy of 80%.
- Class 1 (survived) was predicted with higher precision and recall compared to class 0 (not survived).

---

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `sklearn`
  - `google.colab` (for Google Drive integration)

---

## How to Run the Code
1. Mount Google Drive and ensure the dataset is uploaded to the specified path.
2. Run the Python script in Google Colab or any Python environment with the required libraries installed.
3. Examine the outputs for accuracy, classification report, and confusion matrix.

---

## Conclusion
This project demonstrates the effective use of Logistic Regression for classification tasks. Proper preprocessing and handling of outliers and class imbalance were critical to the model's performance.

