import mlflow
import dagshub
from dotenv import load_dotenv
from credit_pipeline.utils.config import load_config
from credit_pipeline.utils.paths import CONFIG_DIR, MODELS_DIR, SHAP_DIR, FAIRNESS_DIR


def setup_mlflow():
    load_dotenv()
    config = load_config(CONFIG_DIR / "config.yaml")

    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    repo_owner = config["dagshub"]["repo_owner"]
    repo_name = config["dagshub"]["repo_name"]

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_child_run(model_name, params, metrics, model, parent_run_id):
    with mlflow.start_run(
        run_name=f"child_{model_name}", nested=True, parent_run_id=parent_run_id
    ):
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "auc_pr": metrics["auc_pr"],
                "auc_roc": metrics["auc_roc"],
                "f2": metrics["f2"],
            }
        )
        mlflow.sklearn.log_model(model, model_name)


def log_parent_run(metrics_by_threshold, models):
    for threshold, metrics in metrics_by_threshold.items():
        mlflow.log_metrics(
            {
                f"auc_pr_{threshold}": metrics["auc_pr"],
                f"auc_roc_{threshold}": metrics["auc_roc"],
                f"f2_{threshold}": metrics["f2"],
            }
        )

    # Log individual model artifacts
    for model_name in ["xgb_best", "lgb_best", "catboost_best", "fitted_transformer"]:
        mlflow.log_artifact(MODELS_DIR / f"{model_name}.joblib")

    # Log plot artifacts
    mlflow.log_artifact(MODELS_DIR / "pr_curve.png")
    mlflow.log_artifact(MODELS_DIR / "roc_curve.png")

    # Log artifact directories
    mlflow.log_artifacts(str(SHAP_DIR), artifact_path="shap")
    mlflow.log_artifacts(str(FAIRNESS_DIR), artifact_path="fairness")


# def register_model(run_id, registered_model_name):
#     model_uri = f"runs:/{run_id}/Ensemble"
#     mlflow.register_model(model_uri=model_uri, name=registered_model_name)

#     client = mlflow.tracking.MlflowClient()
#     versions = client.get_latest_versions(registered_model_name, stages=["None"])
#     if versions:
#         client.transition_model_version_stage(
#             name=registered_model_name, version=versions[0].version, stage="Production"
#         )
