import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from credit_pipeline.preprocessing.pipeline import build_pipeline
# from credit_pipeline.utils.config import load_config


logger = logging.getLogger(__name__)


def evaluate_model(model_name, y_test, y_proba, threshold):
    # Convert probabilities to binary predictions at given threshold
    y_pred = (y_proba >= threshold).astype(int)

    auc_pr = average_precision_score(y_test, y_proba)
    auc_roc = roc_auc_score(y_test, y_proba)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"\n--- {model_name} @ threshold={threshold} ---")
    logger.info(f"AUC-PR:  {auc_pr:.4f}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"F2:      {f2:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")

    return {
        "model": model_name,
        "threshold": threshold,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "f2": f2,
        "confusion_matrix": cm,
    }


def plot_precision_recall(models_probas, y_test, save_path="models/pr_curve.png"):
    plt.figure(figsize=(8, 6))
    for model_name, y_proba in models_probas.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, label=f"{model_name} (AUC-PR={auc_pr:.3f})")

    plt.axhline(y=0.08, color="gray", linestyle="--", label="Random baseline (0.08)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"PR curve saved to {save_path}")


def plot_roc_curve(models_probas, y_test, save_path="models/roc_curve.png"):
    plt.figure(figsize=(8, 6))
    for model_name, y_proba in models_probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_roc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC-ROC={auc_roc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"ROC curve saved to {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = "E:\\DCS Final Project\\credit-ml-pipeline\\data\\application_data.csv"
    config_path = (
        "E:\\DCS Final Project\\credit-ml-pipeline\\config\\preprocessing_config.yaml"
    )
    prev_path = (
        "E:\\DCS Final Project\\credit-ml-pipeline\\data\\previous_application.csv"
    )

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers = (
        build_pipeline(data_path, prev_path, config_path)
    )

    # Load models
    xgb_best = joblib.load("models/xgb_best.joblib")
    lgb_best = joblib.load("models/lgb_best.joblib")
    catboost_best = joblib.load("models/catboost_best.joblib")

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

    # Evaluate at both thresholds
    for model_name, y_proba in models_probas.items():
        for threshold in [0.3, 0.5]:
            evaluate_model(model_name, y_test, y_proba, threshold)

    # Plot PR and Roc curves
    plot_precision_recall(models_probas, y_test)
    plot_roc_curve(models_probas, y_test, save_path="models/roc_curve.png")
