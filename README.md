# End-to-End Sales Forecasting & Demand Intelligence System

📌 Project Overview

The End-to-End Sales Forecasting & Demand Intelligence System is a comprehensive Data Science project designed to help retail and e-commerce businesses make data-driven inventory and sales decisions. The system predicts future sales demand, detects unusual sales patterns, segments products based on demand behavior, and provides an interactive dashboard for business users.

This project combines Time Series Analysis, Machine Learning, Anomaly Detection, Clustering, and Business Intelligence into a single deployable application.

🎯 Problem Statement

Retail and e-commerce companies constantly face the challenge of predicting future product demand. Overstocking increases storage costs and locks working capital, while understocking results in lost sales and dissatisfied customers.

This project addresses these challenges by building an intelligent forecasting system that:

Predicts future product sales
Detects unusual sales spikes and drops
Segments products based on demand characteristics
Provides actionable inventory recommendations
Visualizes insights through an interactive Streamlit dashboard
🚀 Features
📊 Sales Analysis
Sales trend analysis
Annual and monthly sales visualization
Region-wise sales analysis
Category-wise sales analysis
Interactive filters
📈 Sales Forecasting

Implemented and compared three forecasting approaches:

SARIMA
Facebook Prophet
XGBoost Regressor

Performance evaluation using:

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)

The best-performing model (XGBoost) is used for future forecasting.

🚨 Anomaly Detection

Two anomaly detection techniques were implemented:

Isolation Forest
Rolling Z-Score

The system identifies:

Unusual sales spikes
Unexpected sales drops
Potential festive sales
Inventory-related anomalies
📦 Product Demand Segmentation

Product sub-categories are segmented using K-Means Clustering based on:

Total Sales
Sales Growth Rate
Sales Volatility
Average Order Value

Demand clusters include:

High Volume, Stable Demand
Growing Demand
Low Volume, High Volatility
Declining Demand

Each cluster includes business-oriented stocking recommendations.

🌐 Interactive Streamlit Dashboard

The project includes a fully interactive Streamlit application with four modules:

📊 Sales Overview
KPI Cards
Annual Sales
Monthly Sales Trend
Region Filter
Category Filter
📈 Forecast Explorer
Category/Region Selection
Forecast Horizon (1–3 Months)
XGBoost Sales Forecast
MAE & RMSE Display
🚨 Anomaly Report
Isolation Forest Visualization
Rolling Z-Score Visualization
Anomaly Table
Downloadable Report
📦 Product Demand Segments
PCA Cluster Visualization
Demand Cluster Table
Inventory Recommendations
🛠️ Technologies Used
Programming Language
Python
Data Analysis
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Machine Learning
Scikit-learn
XGBoost
Time Series Forecasting
Statsmodels (SARIMA)
Facebook Prophet
Anomaly Detection
Isolation Forest
Rolling Z-Score
Clustering
K-Means
PCA
Dashboard
Streamlit
📊 Dataset
Primary Dataset

Superstore Sales Dataset

Contains:

Order Details
Customer Information
Product Categories
Sales
Regions
Shipping Information
Supplementary Dataset

Video Game Sales Dataset

Used for:

Isolation Forest Anomaly Detection
Comparative anomaly analysis
📈 Model Performance
Model	MAE	RMSE	MAPE
SARIMA	20,580.71	22,190.93	21.94%
Prophet	20,250.79	22,318.41	21.86%
XGBoost	15,009.50	20,566.48	14.10%

Best Model: ✅ XGBoost Regressor

📌 Business Insights
Technology products show strong future demand.
Office Supplies generate the highest forecasted sales.
Holiday periods produce significant sales spikes.
Isolation Forest successfully detects both sales spikes and sales drops.
Product segmentation enables smarter inventory planning and replenishment.
💻 Installation

Clone the repository:

git clone https://github.com/pakshalsolanki05/SalesForecasting_Pakshal.git

Move into the project directory:

cd SalesForecasting_Pakshal

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py
🌍 Live Demo

Streamlit App:
Add your deployed Streamlit Community Cloud URL here after deployment.

Example:

https://salesforecasting-pakshal.streamlit.app
📸 Project Screenshots

Add screenshots of:

Sales Overview Dashboard
Forecast Explorer
Anomaly Report
Demand Segmentation
🔮 Future Enhancements
Deep Learning forecasting using LSTM/GRU
Real-time sales prediction
Inventory optimization
Price elasticity analysis
Demand forecasting at product level
Automated report generation
Cloud database integration
REST API deployment

👨‍💻 Author
Pakshal

Artificial Intelligence & Data Science Engineering Student

⭐ If you found this project useful

Please consider giving this repository a ⭐ Star to support the project and make it easier for others to discover.
