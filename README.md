# 🏠 Real Estate Investment Advisory System

An end-to-end Machine Learning application that assists potential investors in evaluating real estate properties by:
1. **Classifying whether a property is a good investment**
2. **Predicting the estimated property price after 5 years**

The system is built with a production-first mindset, featuring structured preprocessing, feature engineering, multi-model comparison using MLflow, and deployment via Streamlit.

---

## 🚀 Key Features

- **Investment Classification**
  - Predicts whether a property qualifies as a *Good Investment*
  - Uses a composite, rule-informed target engineered from market-relevant signals

- **5-Year Price Forecasting**
  - Predicts future property price using regression models
  - Incorporates city-tier–based growth assumptions

- **Robust ML Pipeline**
  - Deterministic preprocessing and feature engineering
  - Multiple models trained and compared (Logistic, Random Forest, XGBoost, Linear Regression)

- **MLflow Integration**
  - Tracks experiments, metrics, and model artifacts
  - Enables transparent model selection

- **Interactive Streamlit App**
  - User-friendly interface for real-time predictions
  - Reuses the exact training pipeline for inference (no data leakage)

---

## 🧠 Problem Formulation

### 1. Classification Task
**Target:** `Good_Investment`  
**Objective:** Decide whether a property is worth buying from an investment perspective.

The target is **engineered**, not observed, using a composite investment score derived from:
- Price per SqFt (value vs city benchmark)
- Public transport accessibility
- Amenities
- Property age
- Nearby schools

This score is thresholded to ensure class balance and real-world interpretability.

---

### 2. Regression Task
**Target:** `Future_Price_5Y`  
**Objective:** Predict the estimated property price after 5 years.

The forecast is based on:
- Current price
- City tier (Tier 1 / Tier 2 / Tier 3)
- Property characteristics

This reflects realistic analyst-style forecasting rather than historical time-series modeling.

---

## 🔬 Exploratory Data Analysis (EDA)

EDA was conducted in four structured phases:

1. **Price & Size Analysis**
2. **Location-Based Analysis**
3. **Feature Relationships & Correlations**
4. **Investment, Amenities & Ownership Analysis**

Key findings:
- Pricing is intentionally weakly correlated with individual features
- Investment quality emerges only through **multi-factor reasoning**
- Justifies tree-based models and composite scoring

---

## 🤖 Models Trained

### Classification
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

**Primary metric:** ROC AUC  
**Secondary focus:** Recall for positive class

---

### Regression
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor

**Primary metric:** RMSE  
**Secondary metric:** MAE

---

## 📊 Experiment Tracking with MLflow

- All models trained under controlled conditions
- Metrics and artifacts logged to MLflow
- Best-performing models manually selected and persisted for deployment

---

## 🧪 Feature Engineering Highlights

- `Age_of_Property` derived from `Year_Built`
- `Price_per_SqFt` recomputed to avoid hidden leakage
- `Amenity_Count` extracted from raw amenity strings
- `Transport_Score` encoded as an ordinal feature
- `City_Tier` applied consistently across training and inference

---

## 🖥️ Streamlit Application

The Streamlit app:
- Collects user inputs via a form
- Applies identical preprocessing and feature engineering
- Outputs:
  - Investment decision with confidence
  - Estimated 5-year future price

### Run the app:
```bash
streamlit run app.py

