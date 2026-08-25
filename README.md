# 🛒 E-Commerce Customer Churn Prediction & Retention Engine

An end-to-end Machine Learning solution designed to detect at-risk customers and mitigate churn in e-commerce platforms. Built with a **Random Forest Classifier** and deployed as an interactive, real-time web application using **Streamlit**.

---

## 📌 Project Overview

Customer Acquisition Cost (CAC) is significantly higher than Customer Retention Cost. This project provides businesses with an automated decision-support system to identify high-risk churn patterns early—especially "silent churners"—and trigger proactive retention strategies to maximize Customer Lifetime Value (CLV).

The project follows the standard **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:
1. **Business Understanding:** Mitigating churn to protect CLV.
2. **Data Understanding & EDA:** Uncovering behavioral drivers of attrition across 50,000+ records.
3. **Data Preparation & Feature Engineering:** Encoding, scaling, and handling multi-variable behavioral metrics.
4. **Modeling:** Training and tuning an ensemble Random Forest model.
5. **Evaluation:** Prioritizing Recall (0.92) to minimize missed churners.
6. **Deployment:** Interactive Streamlit dashboard for real-time predictions.

---

## 🚀 Key Features

- **High-Sensitivity Predictive Engine:** Evaluates multi-dimensional customer behavioral signals (call frequency, session time, cart abandonment, login frequency, tenure, and spend).
- **Proactive Retention Recommendations:** Automatically provides actionable next steps (e.g., targeted discounts, priority outreach) based on risk scores.
- **Interactive UI:** A lightweight web dashboard built with Streamlit allowing sales and retention teams to input customer data and get instant results.
- **Explainable Insights:** Leverages feature importance metrics to pinpoint root causes of churn.

---

## 📊 Model Performance & Metrics

Evaluated on a holdout test set of 10,000 customer records:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **91.53%** |
| **Recall (Churn Class)** | **0.92** |
| **True Negatives (Stayed)** | 6,946 |
| **True Positives (Churn Caught)** | 2,207 |
| **False Positives** | 184 |
| **False Negatives** | 663 |

> **Note on Strategy:** In churn management, **Recall** is prioritized over Precision. Missing an at-risk customer (False Negative) is substantially costlier to the business than offering an unnecessary retention perk to a loyal customer (False Positive).

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python
- **Machine Learning:** Scikit-Learn
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib / Pickle
- **Web App / UI:** Streamlit

---
