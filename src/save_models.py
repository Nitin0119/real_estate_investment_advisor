import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import src.model_loader
from src.model_loader import load_model_from_run
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# -------------------------
# GLOBAL CONFIG
# -------------------------

CLASSIFIER_EXPERIMENT = "real_estate_classifier"
REGRESSOR_EXPERIMENT = "real_estate_regressor"

CLASSIFIER_METRIC = "roc_auc"   # higher is better
REGRESSOR_METRIC = "rmse"       # lower is better

MODEL_DIR = "models"


def get_best_run_id(experiment_name: str, metric: str, ascending: bool) -> str:
    """
    Fetch best run_id from an experiment based on a metric.
    """
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    order = "ASC" if ascending else "DESC"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found for experiment: {experiment_name}")

    return runs[0].info.run_id


def save_model():

    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # -------------------------
    # Resolve best runs
    # -------------------------
    classifier_run_id = get_best_run_id(
        experiment_name=CLASSIFIER_EXPERIMENT,
        metric=CLASSIFIER_METRIC,
        ascending=False,  # higher ROC-AUC is better
    )

    regressor_run_id = get_best_run_id(
        experiment_name=REGRESSOR_EXPERIMENT,
        metric=REGRESSOR_METRIC,
        ascending=True,   # lower RMSE is better
    )

    print("Best classifier run:", classifier_run_id)
    print("Best regressor run :", regressor_run_id)

    # -------------------------
    # Load models
    # -------------------------
    clf = load_model_from_run(classifier_run_id)
    reg = load_model_from_run(regressor_run_id)

    # -------------------------
    # Persist locally
    # -------------------------
    joblib.dump(clf, f"{MODEL_DIR}/investment_classifier.pkl")
    joblib.dump(reg, f"{MODEL_DIR}/investment_regressor.pkl")

    print("✅ Models saved successfully")


if __name__ == "__main__":
    save_model()
