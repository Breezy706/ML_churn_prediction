# 🏦 Bank Customer Churn Prediction using Machine Learning

**DATA SCIENTIST:** BREEZY_YRN  

---

## 📌 Project Overview

Customer retention is one of the biggest challenges in the banking sector. Acquiring new customers is significantly more expensive than retaining existing ones.  

This project focuses on **predicting customer churn**, where churn refers to customers who discontinue their relationship with the bank.

Using machine learning techniques, this project:
- Analyzes customer behavior
- Identifies key churn factors
- Builds predictive models to detect high-risk customers

💡 The goal is to help banks:
- Improve customer retention strategies  
- Increase customer satisfaction  
- Reduce revenue loss  

---

## 📂 Dataset Information

The dataset contains **10,000 customer records** with 14 features, including:

- Demographic data (Age, Gender, Geography)
- Financial data (Balance, Credit Score, Salary)
- Behavioral data (Tenure, Products, Activity)
- Target variable: **Exited (0 = No, 1 = Yes)**
---

## 🔧 Technologies Used

- Python 🐍  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Scikit-learn  
- Imbalanced-learn (SMOTEENN)  
- XGBoost  

---

## 📊 Exploratory Data Analysis (EDA)

Key steps performed:
- Data cleaning (removed irrelevant columns)
- Handling categorical variables (encoding)
- Feature engineering (Age grouping)
- Data visualization (univariate & bivariate analysis)

### 🔍 Key Insights

- 📉 Churn rate ≈ **20%** (imbalanced dataset)
- 🌍 Germany has the highest churn rate
- 👩 Female customers churn more than males
- 👥 Middle-aged customers (30–53) are more likely to churn
- 💳 Customers with **1 product** churn the most
- ⚠️ Inactive members are highly likely to churn
- 💰 Higher balance customers show higher churn tendency

---

## 🧹 Data Preprocessing

- Removed unnecessary columns:
  - RowNumber, CustomerId, Surname  
- Applied **One-Hot Encoding**
- Feature scaling using **StandardScaler**
- Train-Test split (80/20)

---

## ⚖️ Handling Imbalanced Data

The dataset is imbalanced (80:20 ratio).  

To solve this:
- Applied **SMOTEENN (Oversampling + Undersampling)**

---

## 🤖 Machine Learning Models Used

The following models were implemented:

- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Logistic Regression  
- XGBoost  

---

## 📈 Model Performance

| Model | Accuracy | Notes |
|------|--------|------|
| Decision Tree | ~85% | Low recall on churn |
| Random Forest | ~86% | Better performance |
| KNN | ~81% | Moderate |
| Naive Bayes | ~81% | Weak recall |
| SVM | ~85% | Balanced |
| Logistic Regression | ~82% | Baseline |
| **XGBoost (Best)** | **~87%** | Highest recall & F1-score |

---

## 🏆 Best Model

✅ **XGBoost with SMOTEENN** achieved the best results:

- Accuracy: **~87%**
- High recall for churn detection
- Balanced precision and F1-score

---

## 🧠 Feature Importance Insights

Top factors influencing churn:
- Age (especially 42–53 group)
- Geography (Germany)
- Customer activity
- Balance
- Number of products

---

## 🔬 Dimensionality Reduction

- Applied **PCA (90% variance)**
- Result: No performance improvement  
👉 Final model used **without PCA**
