# Predictive Analytics Mini Project — Full Data Cleaning

## Objective
Predict Pass/Fail from study hours, attendance percentage and previous score.

## One ML model only
Logistic Regression (binary classification). This is not a time-series project.

## Cleaning process included
1. Load and inspect data: shape, preview, data types, missing counts, duplicates and descriptive statistics.
2. Validate data: attendance must be between 0 and 100. Invalid values are converted to missing.
3. Remove exact duplicate rows.
4. Detect outliers using IQR: lower bound = Q1 − 1.5×IQR; upper bound = Q3 + 1.5×IQR.
5. Cap outliers to the IQR bounds rather than deleting the records.
6. Split into training and test sets (75:25).
7. Impute missing feature values using the training-set median.
8. Standardize features using StandardScaler.
9. Train Logistic Regression and evaluate accuracy, F1-score, ROC-AUC and confusion matrix.
10. Generate EDA/evaluation plots and an example prediction.

The imputer and scaler are fitted only on training data to avoid data leakage.

## Dataset
`data/student_performance_raw.csv` intentionally contains missing values, one duplicate row, invalid attendance (180%) and extreme values. This lets the full cleaning workflow be demonstrated.

## Run on macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

See `outputs/full_process_log.txt` for the step-by-step record after running.

## Limitation
The data is synthetic and small, for coursework demonstration only. Model scores are not evidence of real-world performance.
