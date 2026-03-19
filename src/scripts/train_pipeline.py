import logging
import mlflow
import optuna

from credit_pipeline.preprocessing.pipeline import build_pipeline
from credit_pipeline.models.train import model_training
from credit_pipeline.models.ensemble import ensemble_predict
from credit_pipeline.models.evaluate import (
    evaluate_model,
    plot_precision_recall,
    plot_roc_curve,
)
from credit_pipeline.utils.config import load_config
from credit_pipeline.utils.paths import CONFIG_DIR, DATA_DIR, MODELS_DIR, OPTUNA_DB_PATH
from credit_pipeline.utils.mlflow_logger import (
    setup_mlflow,
    log_child_run,
    log_parent_run,
)

logger = logging.getLogger(__name__)

THRESHOLDS = [0.3, 0.5, 0.68]


def run_pipeline():
    logging.basicConfig(level=logging.INFO)

    # Paths
    data_path = DATA_DIR / "application_data.csv"
    prev_path = DATA_DIR / "previous_application.csv"
    config_path = CONFIG_DIR / "preprocessing_config.yaml"
    main_config_path = CONFIG_DIR / "config.yaml"

    config = load_config(main_config_path)
    registered_model_name = config["mlflow"]["registered_model_name"]
    storage = f"sqlite:///{OPTUNA_DB_PATH}"

    # 1. Build pipeline
    X_train, X_test, y_train, y_test, *_ = build_pipeline(
        data_path, prev_path, config_path
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # 2. Setup MLflow
    setup_mlflow()

    # 3. Open parent run
    with mlflow.start_run(run_name="parent_Ensemble") as parent_run:
        parent_run_id = parent_run.info.run_id

        # 4. Train models + log child runs
        xgb_best, lgb_best, catboost_best = model_training(
            X_train, X_test, y_train, y_test, scale_pos_weight, storage
        )

        # Load best params from Optuna for logging
        for model_name, study_name, model in [
            ("XGBoost", "xgb_study", xgb_best),
            ("LightGBM", "lightGBM_study", lgb_best),
            ("CatBoost", "catboost_study", catboost_best),
        ]:
            study = optuna.load_study(study_name=study_name, storage=storage)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(model_name, y_test, y_proba, threshold=0.5)
            log_child_run(model_name, study.best_params, metrics, model, parent_run_id)

        # 5. Ensemble
        ensemble_proba, _ = ensemble_predict(X_test, y_test)

        # 6. Evaluate ensemble at all thresholds
        metrics_by_threshold = {}
        for threshold in THRESHOLDS:
            metrics_by_threshold[threshold] = evaluate_model(
                "Ensemble", y_test, ensemble_proba, threshold
            )

        # 7. Generate and save plots
        models_probas = {
            "XGBoost": xgb_best.predict_proba(X_test)[:, 1],
            "LightGBM": lgb_best.predict_proba(X_test)[:, 1],
            "CatBoost": catboost_best.predict_proba(X_test)[:, 1],
            "Ensemble": ensemble_proba,
        }
        plot_precision_recall(models_probas, y_test, MODELS_DIR / "pr_curve.png")
        plot_roc_curve(models_probas, y_test, MODELS_DIR / "roc_curve.png")

        # 8. Log parent run
        log_parent_run(metrics_by_threshold, models_probas)

        # 9. Log ensemble model + register
        mlflow.sklearn.log_model(
            {"xgb": xgb_best, "lgb": lgb_best, "cat": catboost_best},
            artifact_path="Ensemble",
        )
        mlflow.end_run(status="FINISHED")
        # register_model(parent_run_id, registered_model_name)

        logger.info(f"Pipeline complete. Parent run ID: {parent_run_id}")


if __name__ == "__main__":
    run_pipeline()
