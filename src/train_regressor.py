import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import TARGET_REGRESSION
from src.preprocessing import preprocess_base
from src.feature_engineering import add_engineered_features
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "real_estate_regressor"
NAME = "model"

def train_regressor(data_path: str):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path)
    df = preprocess_base(df)
    df = add_engineered_features(df)

    x = df.drop(columns=[TARGET_REGRESSION, 'good_investment'])
    y = df[TARGET_REGRESSION]

    cat_features = ['city_tier', "property_type"]

    num_features = [
        "price_in_lakhs", "price_per_sqft", "size_in_sqft",
        "age_of_property", "transport_score", "bhk"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('num', "passthrough", num_features),
        ]
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            random_state=42,
            n_jobs=-1),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    for model_name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        try:
            with mlflow.start_run(run_name=f"regressor_{model_name}"):
                pipeline.fit(X_train, y_train)

                preds = pipeline.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

                mlflow.log_param("model_name", model_name)
                mlflow.log_param("target", TARGET_REGRESSION)

                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                mlflow.sklearn.log_model(pipeline, name=NAME)

                print(f"\n=== {model_name.upper()} ===")
                print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        except Exception as e:
            print(f"\n❌ {model_name} FAILED")
            print(e)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )[0]
    best_run_id = best_run.info.run_id

    print("\n✅ Best regressor selected")
    print(f"Run ID : {best_run_id}")
    print(f"RMSE   : {best_run.data.metrics['rmse']}")

    return best_run_id