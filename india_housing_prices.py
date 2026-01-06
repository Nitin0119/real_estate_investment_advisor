# Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load & Clean Data
df = pd.read_csv("india_housing_prices.csv")
df = df.drop_duplicates()

# Drop identifiers / leakage-prone columns if present
drop_cols = [c for c in ["ID", "Price_per_SqFt"] if c in df.columns]
df = df.drop(columns=drop_cols)

# Defensive Data Cleaning
# Size == 0 is invalid data, not a math edge case
df["Size_in_SqFt"] = df["Size_in_SqFt"].replace(0, np.nan)
df = df.dropna(subset=["Size_in_SqFt", "Price_in_Lakhs", "Year_Built"])

# Feature Engineering
CURRENT_YEAR = 2026

df["Price_per_SqFt"] = (
    df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
)

df["Property_Age"] = CURRENT_YEAR - df["Year_Built"]

# Robust city baseline (median, not mean)
df["City_Median_PPSF"] = (
    df.groupby("City")["Price_per_SqFt"].transform("median")
)

df["Relative_Price_Index"] = (
    df["Price_per_SqFt"] / df["City_Median_PPSF"]
)

# Target Definition (POC logic)
df["good_investment"] = (
    (df["Relative_Price_Index"] < 1) &
    (df["Property_Age"] <= 10)
).astype(int)

# Feature Selection
# Drop City from categorical encoding
# City signal is already embedded in engineered features
num_cols = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Property_Age",
    "Relative_Price_Index"
]

cat_cols = [
    "State",
    "Property_Type",
    "Furnished_Status",
    "Facing",
    "Owner_Type",
    "Availability_Status"
]

X = df[num_cols + cat_cols]
y = df["good_investment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols)
    ]
)

# Model Pipeline
pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]
)

# Hyperparameter Tuning
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear"]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Best Params:", grid.best_params_)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
