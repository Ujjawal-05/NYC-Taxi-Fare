## 🚕 NYC Taxi Fare Prediction
This project aims to predict taxi fares in New York City using a variety of machine learning models. It is based on the popular Kaggle competition. The primary objective is to build and compare regression models to accurately predict the fare amount given features like pickup and drop-off locations, date/time, and passenger count.

---

## Competiton Link :- https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction

---

## 📌 Project Overview
### Dataset
- The dataset contains millions of taxi trip records including:
- Pickup and drop-off datetime
- Pickup and drop-off latitude/longitude
- Passenger count
- Fare amount (target variable)

### Goal
Build a regression model that can predict the fare amount given a set of input features.

---

## 📁 File Structure
- NYC_Taxi_Fare.ipynb — Main Jupyter notebook with data processing, feature engineering, model training, and evaluation.
- xgb_tuned_submission.csv — Submission file using Tuned XGBoost.
- XGBReg.csv — Submission file using default XGBRegressor.
- RandomForest.csv — Submission file using Random Forest Regressor.
- ridge_submission.csv — Submission file using Ridge Regression.
- README.md — Project documentation.

---

## 🔍 Data Preprocessing and Feature Engineering
- The following steps were used:
- Removed outliers based on lat/lon bounds and fare ranges
- Extracted temporal features from pickup_datetime:
- Hour, Day, Month, Day of Week
- Calculated Haversine distance and Manhattan distance between pickup and drop-off coordinates
- Standardized numerical features using StandardScaler

---

## 📊 Models Used and Results

| Model                | Public Score | Private Score |
|----------------------|--------------|---------------|
| Tuned XGBoost        | 3.23044      | 3.23044       |
| XGBRegressor (Base)  | 3.31657      | 3.31657       |
| Random Forest        | 3.35896      | 3.35896       |
| Ridge Regression     | 5.15759      | 5.15759       |

✅ **Best Model:** Tuned XGBoost  
📉 **Metric Used:** Root Mean Squared Error (RMSE)

---

## ⚙️ Model Tuning
- XGBoost: Hyperparameters tuned using GridSearchCV and manual experimentation:
  - n_estimators, learning_rate, max_depth, subsample, colsample_bytree
- Random Forest: Tried different n_estimators and max_depth
- Ridge Regression: Tuned alpha with cross-validation

---

## 📈 Future Improvements
- Implement deep learning with TensorFlow/Keras  
- Use feature selection techniques  
- Add weather and traffic data  
- Use ensemble stacking or blending  

---

## 🧠 Key Learnings
- How to work with geospatial data  
- Importance of feature engineering in tabular data  
- Model evaluation using Kaggle's public/private leaderboard split  
- Hyperparameter tuning and model comparison  

---

## 📤 Submission
Submissions were made on Kaggle after the competition deadline for learning purposes.  
The `xgb_tuned_submission.csv` achieved the best RMSE.

---

## Result Snapshot
<img width="1464" height="513" alt="Screenshot 2025-07-19 124932" src="https://github.com/user-attachments/assets/5585499d-57ee-4253-a1d7-b3f2e3da769b" />
