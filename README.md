# Blinkit-Sales-Analysis
End-to-end data analysis project using SQL, Python, and Power BI

# Blinkit Sales Analysis & Business Intelligence Project


## Project Overview

This project presents an **end-to-end data analysis pipeline** on Blinkit retail data, combining **data cleaning, SQL analysis, machine learning, and Power BI visualization** to derive actionable business insights.

The goal was to simulate a **real-world analytics scenario** where business stakeholders can make **data-driven decisions** regarding sales performance, outlet optimization, and product strategy.


## Tools & Technologies Used

* **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
* **SQL (MySQL)** for data querying & analysis
* **Power BI** for interactive dashboard
* **Excel** for preprocessing & feature engineering


## Dashboard Overview

The Power BI dashboard provides a **comprehensive view of sales performance**, covering:

* KPI metrics (Total Sales, Avg Sales, Total Items, Avg MRP)
* Product category performance
* Outlet-level analysis
* City-wise sales distribution (Map visualization)
* Sales trends over time


## Dashboard Preview

<img width="1306" height="733" alt="image" src="https://github.com/user-attachments/assets/7a0e6fa3-e863-4a28-bab1-786845b23ec5" />





## Key Business Insights

### Outlet Performance

* **Supermarket Type3** is the highest revenue-generating outlet type
* **Grocery Stores significantly underperform**, indicating operational inefficiencies
* **Tier 3 locations contribute the highest sales**, showing strong demand in emerging markets


### Product Insights

* **Fruits & Snacks categories dominate sales**, making them key revenue drivers
* Product-level distribution indicates **uneven performance across categories**


### Geographic Insights

* Sales are **concentrated in major cities**, with emerging cities showing growth potential
* Geographic segmentation helps identify **expansion opportunities**


### Pricing Impact

* From feature importance analysis:

  * **Item MRP is the most influential factor (~40%)**
  * Pricing strategy plays a critical role in revenue generation


## Machine Learning Models

Three regression models were built to predict **Item Outlet Sales**:

| Model             | MAE    | R² Score |
| ----------------- | ------ | -------- |
| Linear Regression | 944.58 | 0.40     |
| Random Forest     | 756.37 | 0.56     |
| XGBoost           | 748.82 | 0.57     |


Model Insights

* **XGBoost performed best**, achieving the highest R² (~0.57)
* Tree-based models outperform linear models → indicates **non-linear relationships**
* Model performance shows **moderate predictive capability**, leaving room for further tuning


## Feature Importance (Random Forest)

Top influencing features:

* **Item MRP (~40%)**
* Outlet Type (Type1, Type3)
* Item Visibility
* Outlet Establishment Year

**Insight:**

> Pricing and outlet characteristics are the strongest drivers of sales performance

## Business Recommendations

* Expand **Supermarket Type3 outlets** in high-performing regions
* Improve or restructure **Grocery Store operations**
* Focus inventory on **high-performing categories (Fruits, Snacks)**
* Optimize **pricing strategy** based on demand patterns
* Target expansion in **Tier 2 & Tier 3 cities**

## Conclusion

This project demonstrates:

1. End-to-end data analytics workflow
2. Business-driven insights generation
3. Integration of SQL, Python, and BI tools
4. Predictive modeling for decision support


## Author

**Aryan Yuvraj Verma**
Aspiring Data Analyst | Business Analyst

