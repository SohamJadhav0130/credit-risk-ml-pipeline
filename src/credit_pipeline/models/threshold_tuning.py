import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    fbeta_score,
)
from credit_pipeline.preprocessing.pipeline import build_pipeline

# from credit_pipeline.utils.config import load_config
from credit_pipeline.utils.paths import CONFIG_DIR, DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class ThresholdTuner:
    def __init__(self):
        pass

    def find_optimal_threshold(self, y_test, y_proba, model_name):
        thresholds = np.arange(0.1, 0.9, 0.01)
        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            results.append(
                {
                    "threshold": round(threshold, 2),
                    "precision": round(
                        precision_score(y_test, y_pred, zero_division=0), 4
                    ),
                    "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                    "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                    "f2": round(
                        fbeta_score(y_test, y_pred, beta=2, zero_division=0), 4
                    ),
                }
            )
        results_df = pd.DataFrame(results)
        best_f2 = results_df.loc[results_df["f2"].idxmax()]
        best_f1 = results_df.loc[results_df["f1"].idxmax()]
        logger.info(f"\n=== {model_name} Optimal Threshold (F2) ===")
        logger.info(
            f"Threshold: {best_f2['threshold']} | Precision: {best_f2['precision']} | Recall: {best_f2['recall']} | F2: {best_f2['f2']}"
        )
        logger.info(f"\n=== {model_name} Optimal Threshold (F1) ===")
        logger.info(
            f"Threshold: {best_f1['threshold']} | Precision: {best_f1['precision']} | Recall: {best_f1['recall']} | F1: {best_f1['f1']}"
        )
        return best_f2, best_f1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = DATA_DIR / "application_data.csv"
    config_path = CONFIG_DIR / "preprocessing_config.yaml"
    prev_path = DATA_DIR / "previous_application.csv"

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers = (
        build_pipeline(data_path, prev_path, config_path)
    )
    # Load models
    xgb_best = joblib.load(MODELS_DIR / "xgb_best.joblib")
    lgb_best = joblib.load(MODELS_DIR / "lgb_best.joblib")
    catboost_best = joblib.load(MODELS_DIR / "catboost_best.joblib")

    # Get probabilities
    xgb_proba = xgb_best.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_best.predict_proba(X_test)[:, 1]
    catboost_proba = catboost_best.predict_proba(X_test)[:, 1]
    ensemble_proba = (xgb_proba + lgb_proba + catboost_proba) / 3

    models_probas = {
        "XGBoost": xgb_proba,
        "LightGBM": lgb_proba,
        "CatBoost": catboost_proba,
        "Ensemble": ensemble_proba,
    }
    Threshold_tuning = ThresholdTuner()

    # find_optimal_threshold --> Still the best is 0.5
    for model_name, y_proba in models_probas.items():
        Threshold_tuning.find_optimal_threshold(y_test, y_proba, model_name)
