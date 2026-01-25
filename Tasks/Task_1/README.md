# Task 01 – Data Cleaning & Exploratory Data Analysis (EDA)

This task focuses on performing **end-to-end data cleaning, preprocessing, and exploratory data analysis (EDA)** on a real-world retail dataset as part of the **Machine Learning Internship at IncodeVision**.

The objective of this task is to understand the dataset thoroughly, identify patterns and trends, and extract meaningful insights through structured analysis and visualization before applying any machine learning models.

---

## 📊 Dataset Used

**Sample Superstore Dataset**

- Retail sales data containing information on orders, customers, products, sales, profit, discounts, regions, and shipping details.
- The dataset represents real-world business data with both profitable and loss-making transactions.

---

## 🧠 Task Objectives

- Inspect and clean raw data
- Prepare the dataset for analysis by fixing data types and removing irrelevant features
- Engineer meaningful features for deeper insights
- Perform exploratory data analysis using visualization techniques
- Identify business trends, patterns, and loss-driving factors

---

## 🔄 Workflow Followed

1. **Data Loading & Inspection**
   - Loaded dataset using Pandas
   - Examined data shape, structure, and summary statistics

2. **Data Cleaning**
   - Checked for missing values and duplicates
   - Removed non-informative identifier columns
   - Corrected date-related data types

3. **Feature Engineering**
   - Extracted order year and month
   - Calculated shipping time in days
   - Computed profit margin
   - Created time-based features for trend analysis

4. **Exploratory Data Analysis (EDA)**
   - Univariate analysis of sales, profit, and discount
   - Bivariate analysis of sales vs profit and discount vs profit
   - Category, sub-category, segment, and region-level analysis
   - Time-based profit trends across regions
   - Identification of loss-contributing categories

---

## 📈 Visualizations Performed

- Correlation heatmap of numerical features
- Distribution plots for:
  - Sales (log-scaled)
  - Profit (KDE)
  - Discount
- Scatter plots:
  - Sales vs Profit
  - Discount vs Profit
- Category and sub-category-wise bar plots
- Segment-wise average profit analysis
- Year-wise and region-wise profit trends
- Pie charts for sales and loss contribution
- Violin plots for regional sales distribution

> Visualizations were kept intuitive and interpretable to clearly communicate insights.

---

## 💡 Key Insights

- High sales do not always guarantee high profit
- Discounts are a major contributor to losses
- Furniture category contributes significantly to losses
- Home Office segment shows the highest average profit
- Profitability varies across regions and time periods
- A small number of high-discount orders drive a large portion of losses

---

## 🛠️ Tools & Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**

---

## 📓 Notebook

- **File:** `Task_1_EDA.ipynb`
- Contains the complete implementation, visualizations, and analysis for this task

---

## ✅ Task Status

✔ **Completed**

---

## 👤 Author

**Abhyudya Sharma**  
Machine Learning Intern  
**IncodeVision**

---

⭐ This task establishes a strong analytical foundation and demonstrates the importance of clean data and exploratory analysis in real-world machine learning workflows.
