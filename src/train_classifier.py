import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import TARGET_CLASSIFICATION
from src.preprocessing import preprocess_base
from src.feature_engineering import add_engineered_features
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "real_estate_classifier"
NAME = "model"

def train_classifier(data_path: str):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path)
    df = preprocess_base(df)
    df = add_engineered_features(df)

    X = df.drop(columns=[TARGET_CLASSIFICATION, 'future_price_5y'])
    y = df[TARGET_CLASSIFICATION]

    cat_features = [
        'city', 'property_type', 'availability_status', 'owner_type', 'parking_space',
        'security', 'furnished_status'
    ]

    num_features = [
        "price_per_sqft", "age_of_property", "amenity_count",
        "transport_score", "size_in_sqft", "bhk",
        "nearby_schools", "nearby_hospitals"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('num', "passthrough", num_features),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight='balanced'
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    for model_name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        try:
            with mlflow.start_run(run_name=f"classifier_{model_name}"):
                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]

                mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("target", TARGET_CLASSIFICATION)
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                mlflow.sklearn.log_model(pipeline, name=NAME)

                print(f"\n=== {model_name.upper()} ===")
                print(classification_report(y_test, y_pred))

        except Exception as e:
            print(f"\n❌ {model_name} FAILED")
            print(e)

    client = MlflowClient()
    exp = client.get_experiment_by_name("real_estate_classifier")

    best_run = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )[0]
    best_run_id = best_run.info.run_id
    print("\n✅ Best classifier selected")
    print(f"Run ID  : {best_run_id}")
    print(f"ROC-AUC : {best_run.data.metrics['roc_auc']}")

    return best_run_id