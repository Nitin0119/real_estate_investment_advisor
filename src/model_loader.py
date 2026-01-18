import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model_from_run(run_id: str, name: str="model"):
    """Load a model form an MLFlow run ID"""
    if not run_id:
        raise ValueError("run_id must be provided")

    model_uri = f"runs:/{run_id}/{name}"
    return mlflow.sklearn.load_model(model_uri)
