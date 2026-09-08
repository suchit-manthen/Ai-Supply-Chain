\# AI-Based Supply Chain Demand Forecasting and Inventory Optimization System



\## 1. Project Title



AI-Based Supply Chain Demand Forecasting and Inventory Optimization System



\## 2. Project Type



Bachelor of Engineering (BE) AIML Project



\## 3. Problem



In supply chain management, future product demand is uncertain.



If demand is underestimated, organizations may face stockouts and loss of customers.



If demand is overestimated, organizations may maintain excess inventory and face higher storage and carrying costs.



The system should use historical sales and supply-chain-related information to forecast future demand and support inventory replenishment decisions.



\## 4. Main Project Flow



Historical Data

→ AI Demand Prediction

→ Inventory Analysis

→ Stock Alerts

→ Reorder Recommendation



\## 5. Main Inputs



The system may use:



\- Historical sales

\- Product information

\- Price

\- Discounts

\- Seasonality

\- Holidays

\- Current inventory

\- Supplier lead time



\## 6. Main Outputs



The system should provide:



\- Future demand forecasts

\- Inventory status

\- Stockout alerts

\- Overstock alerts

\- Reorder recommendations

\- Recommended additional inventory quantity

\- Web-based dashboard for results



\## 7. Machine Learning / Forecasting Models



The project should support comparison of forecasting approaches.



Initial candidate models:



\- Random Forest

\- XGBoost

\- ARIMA / SARIMA

\- LightGBM

\- LSTM (optional advanced model)



The final model must be selected based on experimental evaluation rather than assuming one model is best.



\## 8. Model Evaluation



Forecasting models should be evaluated using:



\- MAE

\- RMSE

\- MAPE



Additional useful metrics may be added later if appropriate.



\## 9. Feature Engineering



Potential features include:



\- Previous sales

\- Lag features

\- Rolling averages

\- Seasonal patterns

\- Price changes

\- Discounts

\- Holidays

\- Day of week

\- Month

\- Other useful time-related features



\## 10. Demand Forecasting



The selected forecasting model should predict future demand for products over a defined forecasting period.



The exact forecasting horizon will be finalized during implementation.



\## 11. Inventory Analysis



The predicted demand should be combined with:



\- Current inventory

\- Supplier lead time

\- Demand variability



The system should identify possible:



\- Stockout situations

\- Overstock situations



\## 12. Replenishment Recommendation



The system should provide recommendations for:



\- When to reorder

\- How much additional inventory may be required



Potential inventory concepts to evaluate during implementation:



\- Safety stock

\- Reorder point

\- Economic Order Quantity (EOQ)



These should be implemented only where they are appropriate for the available data and project scope.



\## 13. Dataset



The primary forecasting dataset is the M5 Walmart Forecasting dataset.



Relevant data includes:



\- Historical product sales

\- Store information

\- Calendar information

\- Selling prices



Inventory and supplier-related information may need to be added separately for the inventory optimization component.



Any simulated or derived inventory/supplier data must be clearly documented as such.



\## 14. Technology Direction



Proposed technologies:



\### Backend

\- Python

\- Django

\- Django REST Framework



\### Database

\- PostgreSQL



\### Machine Learning

\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- XGBoost

\- LightGBM

\- Statistical forecasting libraries

\- LSTM framework if implemented



\### Frontend

\- React

\- Vite

\- Tailwind CSS

\- Charting library



\## 15. Core Architecture



Data Collection

→ Data Preprocessing

→ Feature Engineering

→ Model Development

→ Model Evaluation

→ Demand Forecasting

→ Inventory Analysis

→ Recommendation

→ Web Dashboard



\## 16. Important Development Principle



The project should be implemented incrementally.



Do not implement the entire system at once.



Each module should be developed, tested and verified before moving to the next module.



Do not assume that the most complex or newest model will perform best.



The final forecasting model should be selected based on actual experimental results.



\## 17. Current Scope



The first implementation should focus on:



1\. Understanding the M5 dataset

2\. Preparing clean forecasting data

3\. Creating useful demand-related features

4\. Building baseline forecasting models

5\. Comparing model performance

6\. Selecting a suitable forecasting model

7\. Connecting forecasting with inventory logic

8\. Building the backend API

9\. Building the web dashboard



Advanced features such as explainable AI, probabilistic forecasting, anomaly detection, what-if analysis, hierarchical forecasting, model monitoring and automated retraining may be considered after the core system works.



\## 18. Project Quality Requirements



The final system should prioritize:



\- Correct time-series evaluation

\- Avoidance of data leakage

\- Reproducible experiments

\- Clear separation between ML, backend and frontend

\- Proper API design

\- Database consistency

\- Testable code

\- Clear documentation

