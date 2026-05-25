# ChurnLens — Customer Churn Prediction

> End-to-end Machine Learning pipeline to predict customer churn using the Telco Customer Churn dataset — includes EDA, preprocessing, model training, evaluation, and an interactive web dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

##  Live Demo
 [View Live Dashboard](https://Ujjawal2507.github.io/customer-churn-prediction)
 
## About

**ChurnLens** is a complete Customer Churn Prediction project built with Python and scikit-learn. It analyzes the Telco Customer Churn dataset (7,043 customers, 19 features) to identify customers at risk of leaving — helping businesses take proactive retention action.

---

## What it does

- Performs exploratory data analysis (EDA) with visualizations
- Preprocesses and encodes raw telecom customer data
- Trains and compares 3 ML models — Gradient Boosting, Random Forest, and Logistic Regression
- Auto-selects the best model by AUC-ROC score (~84.5%)
- Scores new customers and segments them into High / Medium / Low churn risk
- Presents all results in a sleek interactive dark-themed web dashboard

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Ujjawal2507/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Get the Telco Customer Churn CSV from Kaggle:
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place it at: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 4. Run EDA
```bash
python src/eda.py
```

### 5. Train and evaluate
```bash
python src/churn_pipeline.py
```

### 6. Predict on new data
```bash
python src/predict.py --input data/new_customers.csv --output outputs/predictions.csv
```

### 7. Open the dashboard
Simply open `dashboard.html` in your browser — no server needed.

---

## Models Compared

| Model | Accuracy | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| Gradient Boosting | 80.2% | 0.845 | 67% | 72% |
| Random Forest | 78.1% | 0.821 | 64% | 68% |
| Logistic Regression | 75.8% | 0.784 | 59% | 61% |

Best model is auto-selected by cross-validated AUC and saved to `outputs/best_model.pkl`.

---

## Key Dataset Features

| Feature | Description |
|---|---|
| `tenure` | Months customer has been with the company |
| `Contract` | Month-to-month, one year, two year |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Cumulative charges |
| `InternetService` | DSL, Fiber optic, or None |
| `PaymentMethod` | Electronic check, mailed check, bank transfer, credit card |
| `Churn` | **Target**: Yes = churned, No = retained |

---

## Output Files

| File | Description |
|---|---|
| `outputs/best_model.pkl` | Trained best model |
| `outputs/scaler.pkl` | Fitted StandardScaler |
| `outputs/features.csv` | Feature column order for inference |
| `outputs/confusion_matrix.png` | Confusion matrix for best model |
| `outputs/roc_curves.png` | ROC curves for all models |
| `outputs/feature_importance.png` | Top 15 features |
| `outputs/eda/*.png` | EDA charts |

---

## Tech Stack

`Python` &nbsp; `scikit-learn` &nbsp; `pandas` &nbsp; `numpy` &nbsp; `matplotlib` &nbsp; `seaborn` &nbsp; `Chart.js` &nbsp; `HTML/CSS/JS`

---

## Author

**Ujjawal** — [github.com/Ujjawal2507](https://github.com/Ujjawal2507)
