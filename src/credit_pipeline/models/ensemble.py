import logging
import joblib
from sklearn.metrics import average_precision_score
from credit_pipeline.preprocessing.pipeline import build_pipeline
from credit_pipeline.utils.paths import CONFIG_DIR, DATA_DIR

logger = logging.getLogger(__name__)


def ensemble_predict(X_test, y_test):
    # Load saved models
    xgb_best = joblib.load("models/xgb_best.joblib")
    lgb_best = joblib.load("models/lgb_best.joblib")
    catboost_best = joblib.load("models/catboost_best.joblib")

    # Get probabilities
    xgb_proba = xgb_best.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_best.predict_proba(X_test)[:, 1]
    catboost_proba = catboost_best.predict_proba(X_test)[:, 1]

    # Average ensemble
    ensemble_proba = (xgb_proba + lgb_proba + catboost_proba) / 3
    ensemble_score = average_precision_score(y_test, ensemble_proba)
    logger.info(f"Ensemble AUC-PR on test set: {ensemble_score}")

    return ensemble_proba, ensemble_score


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = DATA_DIR / "application_data.csv"
    config_path = CONFIG_DIR / "preprocessing_config.yaml"
    prev_path = DATA_DIR / "previous_application.csv"

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers = (
        build_pipeline(data_path, prev_path, config_path)
    )

    ensemble_proba, ensemble_score = ensemble_predict(X_test, y_test)
