# 📊 Task – 04: Customer Churn Prediction Model 📉
**An advanced exploration of Binary Classification, Class Imbalance Handling (SMOTE), and Predictive Analytics**

---

## 📋 Best-in-Class Mission Statement
The objective of this project is to build a **robust Customer Churn Prediction system** capable of identifying customers who are likely to discontinue a service.

By leveraging **customer demographics, service usage patterns, and customer support interactions**, this project delivers a complete **end-to-end classification pipeline** with special emphasis on **class imbalance mitigation** and **recall-optimized model performance**.

This approach reflects real-world business priorities, where **missing a churner is significantly more costly than triggering a false alarm**.

---

## 🔄 Workflow Pipeline

<details>
<summary><b>1. Data Ingestion & Bias Audit 📂</b></summary>
<br>

- 📥 **Dataset Loading:** Imported the `telecom_churn.csv` dataset containing features such as:
  - `CustServCalls`
  - `MonthlyCharge`
  - `DayMinutes`, `InternationalCalls`, and more

- ⚖️ **Class Distribution Analysis:**  
  Identified a strong class imbalance:
  - **Stayed (0):** ~700 customers  
  - **Churned (1):** ~134 customers  

  This confirmed the necessity of imbalance-aware modeling strategies.
</details>

<details>
<summary><b>2. Preprocessing & Feature Engineering ⚙️</b></summary>
<br>

- 🧹 **Feature Selection:**  
  Removed redundant or low-impact attributes such as:
  - `OverageFee`
  - `DataPlan`

- 📐 **Feature Scaling:**  
  Applied `StandardScaler` to normalize numerical variables, ensuring:
  - Improved convergence
  - Stable model performance
</details>

<details>
<summary><b>3. Strategic Bias Handling (SMOTE) 🛡️</b></summary>
<br>

- 🧬 **Synthetic Minority Over-sampling (SMOTE):**  
  Generated synthetic churn samples to balance the training dataset without duplicating existing records.

- ⚖️ **Class Weight Optimization:**  
  Applied:
  - `class_weight='balanced'` for Logistic Regression
  - `scale_pos_weight` for XGBoost  

  These adjustments ensured models focused on learning minority-class behavior.
</details>

<details>
<summary><b>4. Multi-Model Training Engine 🏗️</b></summary>
<br>

- 🧠 **Algorithmic Diversity:**  
  Implemented a reusable `ModelTrainer` class supporting:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Gaussian Naive Bayes

- ✂️ **Train-Test Split:**  
  Split the dataset into training and testing subsets to evaluate real-world generalization.
</details>

<details>
<summary><b>5. Evaluation & Visual Analytics 📉</b></summary>
<br>

- 📏 **Performance Metrics:**  
  Evaluated models using:
  - Recall (primary metric)
  - F1-Score
  - ROC-AUC

- 🎨 **Visual Validation:**  
  Generated:
  - Confusion Matrices
  - ROC Curves  

  The final model achieved an **ROC-AUC score of 0.87**.
</details>

---

## 🧪 Predictive Models & Performance Summary

| Model                | Objective                        | Key Technique Used       | Primary Metric | Performance |
|---------------------|----------------------------------|--------------------------|----------------|-------------|
| Logistic Regression | Baseline Classification          | Class Weight Balancing   | Recall         | 0.72        |
| Random Forest       | Capture Non-Linear Patterns      | Ensemble Learning        | F1-Score       | 0.74        |
| XGBoost             | High-Performance Classification | Boosting + SMOTE         | ROC-AUC        | 0.87        |
| Gaussian Naive Bayes| Probabilistic Benchmark          | Statistical Assumptions  | Recall         | 0.69        |

---

## 🎯 Learning Outcomes

- **Class Imbalance Handling:**  
  Learned to apply SMOTE and class-weighted learning to prevent majority-class dominance.

- **Recall vs Precision Trade-off:**  
  Understood why Recall (0.76) is a mission-critical metric in churn prediction problems.

- **Predictive Analytics:**  
  Transformed historical customer behavior into actionable churn risk predictions.

- **Model Comparison Framework:**  
  Built a modular system for evaluating multiple classifiers consistently.
  
---

## 🚀 Execution Guide

1. **Open** the Jupyter Notebook:  
   `Task_4_Customer_Churn.ipynb`

2. **Run** preprocessing cells to:
   - Clean and scale features
   - Handle imbalance using SMOTE

3. **Execute** the model pipeline:
   - `ModelTrainer`
   - `ModelTester`

4. **Analyze**:
   - Confusion Matrix to validate Recall
   - ROC Curve confirming **0.87 AUC**

---

## 🏁 Final Takeaway
This project demonstrates a **production-ready classification pipeline** where **class imbalance is treated as a core problem**, and model evaluation is driven by **business impact rather than accuracy alone**.
