# 📡 Telecom Customer Churn Analysis & Predictive Modeling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-189fdd.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade machine learning project that analyzes customer behavior, identifies key drivers of subscriber attrition, and deploys high-accuracy classification models with an interactive **Streamlit web application** for proactive retention intervention.

---

## 📌 Executive Summary & Problem Statement

Customer churn represents the rate at which subscribers cancel their service. In the telecommunications sector, acquiring a new customer costs **5× to 7× more** than retaining an existing one. 

This project analyzes **7,043 customer accounts** from the Telco Customer Churn dataset across **21 features** (demographics, services, account contracts, and financial charges) to:
1. Uncover high-risk behavioral churn patterns through rigorous Exploratory Data Analysis (EDA).
2. Engineer high-signal domain features capturing **product stickiness**, **bill shock**, and **retention moats**.
3. Build and evaluate leak-free Machine Learning pipelines (`Logistic Regression`, `Random Forest`, `XGBoost`).
4. Deliver an interactive **Streamlit web app** that scores customer churn probability in real time and recommends actionable retention strategies.

---

## 📊 Key Findings from Exploratory Data Analysis (EDA)

* **The First-Year Critical Window:** **47.4%** of all churn occurs within the first 12 months. Beyond 24 months, churn drops below 20%, and reaches just 6.6% after 5 years.
* **Contract Commitment:** Customers on **Month-to-month** contracts suffer a **42.71%** churn rate and account for **88.6%** of all cancellations. In contrast, 2-year contract customers churn at only **2.83%**.
* **The Security & Support Moat:** Subscribing to `TechSupport` and `OnlineSecurity` cuts churn from **~41.7%** down to **~14.9%**.
* **Service Type Disparity:** Fiber optic users churn at **41.89%** vs. **18.96%** for DSL, driven by high monthly rates ($> \$70/\text{month}$).
* **Payment Friction:** Customers paying via manual **Electronic Check** churn at **45.29%**, compared to only **~15.5%** for automated bank transfers and credit cards.

---

## ⚙️ Advanced Domain Feature Engineering

To capture deeper customer behavior, 8 domain features are engineered with strict train-test separation:

| Engineered Feature | Logic / Calculation | Business & Predictive Value |
| :--- | :--- | :--- |
| **`TotalServicesCount`** | Sum of all 9 active subscribed services | **Product Stickiness:** Multi-product customers face high switching friction and churn at less than half the rate of single-service accounts. |
| **`Protection_Security_Score`** | Count of enrolled protection tools (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`) from 0 to 4 | **Retention Moat:** Churn risk drops monotonically with every added support layer. |
| **`AvgMonthlyCharges`** | $\frac{\text{TotalCharges}}{\text{tenure} + 1}$ | **Lifetime Baseline:** Normalizes customer spend across their entire lifespan. |
| **`Charge_Drift`** | $\text{MonthlyCharges} - \text{AvgMonthlyCharges}$ | **Bill Shock Indicator:** High positive values identify recent rate hikes or expiring promo discounts that trigger churn. |
| **`AutoPayment`** | `1` if Bank Transfer or Credit Card, else `0` | **Frictionless Renewal:** Eliminates manual billing hurdles. |
| **`Is_Long_Term_Contract`** | `1` if 1-year or 2-year contract, else `0` | **Contract Commitment:** Directly isolates at-risk month-to-month contracts. |
| **`Streaming_Only_Risk`** | `1` if Streaming TV/Movies enabled without Security/Support | **High-Cost / Low-Moat:** Identifies accounts paying premium bills without anchoring services. |
| **`Household_Type`** | Categorizes into `Family`, `Couple`, `Single_Parent`, `Solo` | **Life Stage Stability:** Family accounts demonstrate the highest retention stability. |

---

## 🏆 Model Performance & Comparison

Models were trained with stratified 80/20 train-test splits on **1,409 unseen test samples** (1,035 Stays, 374 Churns):

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Recommended)** | 80.41% | 65.51% | **55.35%** | **0.6000** | **0.8432** |
| **Logistic Regression** | **80.91%** | **67.80%** | 53.48% | 0.5979 | **0.8468** |
| **Random Forest** | 79.84% | 64.71% | 52.94% | 0.5824 | 0.8399 |

### Confusion Matrix Breakdown (XGBoost)
* **True Negatives (TN):** `926` loyal customers correctly predicted to stay.
* **False Positives (FP):** `109` customers flagged for review (false alarms).
* **False Negatives (FN):** `167` churners missed by the model at default threshold.
* **True Positives (TP):** `207` churners caught proactively.

> **💡 Business Strategy:** By adjusting the decision threshold from `0.50` down to `0.35`, marketing retention campaigns can boost **Recall to >75%**, capturing the vast majority of churning revenue.

---

## 🚀 Interactive Streamlit Web Application

The project includes an interactive web app (`churn_app.py`) for real-time customer churn evaluation.

### Key Features:
* **Interactive Account Builder:** Input customer tenure, contract terms, internet tiers, and billing details.
* **Real-Time Churn Probability:** Visual risk gauge and percentage metrics.
* **Risk Categorization:** Dynamic tagging (`High Risk`, `Moderate Risk`, `Low Risk`).
* **Prescriptive Action Engine:** Actionable business recommendations tailored to the customer's risk profile.

---

## 📁 Repository Structure

```text
├── Telco_Customer_Churn.csv       # Raw dataset (7,043 rows, 21 columns)
├── churn_app.py                   # Streamlit interactive web application
├── churn_main.py                  # Direct entrypoint for production training
├── churn_model.py                 # Production-grade ML pipeline module
├── churn_pipeline.joblib          # Serialized production pipeline (XGBoost)
├── random_forest_model.joblib     # Serialized Random Forest model artifact
├── model_metadata.json            # Audit trail & performance tracking metrics
├── new.ipynb                      # Complete end-to-end EDA & experimental notebook
├── requirements.txt               # Pinned production environment dependencies
├── .gitignore                     # Git ignore rules for Python & ML artifacts
└── README.md                      # Comprehensive project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/irfan-shekh/Telecom_churn_prediction.git
cd "Telecom_churn_prediction"
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage Guide

### Train the Production Pipeline
Train the default XGBoost pipeline and export new model artifacts:
```bash
python churn_main.py
```

Or customize via CLI options:
```bash
python churn_model.py --model-type xgboost --test-size 0.20 --seed 42
python churn_model.py --model-type random_forest
python churn_model.py --model-type logistic_regression
```

### Launch the Streamlit App
Run the web application locally:
```bash
python -m streamlit run churn_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.
