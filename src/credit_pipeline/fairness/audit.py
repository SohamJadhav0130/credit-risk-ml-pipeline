import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from credit_pipeline.preprocessing.pipeline import build_pipeline
from credit_pipeline.utils.config import load_config
from credit_pipeline.utils.paths import DATA_DIR, CONFIG_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


def fairness_audit(y_test, y_proba, sensitive_col, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "group": sensitive_col.values,
        }
    )

    results = []
    for group, group_df in df.groupby("group"):
        tn, fp, fn, tp = confusion_matrix(
            group_df["y_true"], group_df["y_pred"], labels=[0, 1]
        ).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        pred_positive_rate = (tp + fp) / len(group_df)  # demographic parity

        results.append(
            {
                "group": group,
                "n_samples": len(group_df),
                "n_defaulters": int(group_df["y_true"].sum()),
                "TPR (Recall)": round(tpr, 4),
                "FPR": round(fpr, 4),
                "Precision": round(precision, 4),
                "Pred_Positive_Rate": round(pred_positive_rate, 4),
            }
        )
        logger.info(
            f"Group: {group} | TPR: {tpr:.4f} | FPR: {fpr:.4f} | "
            f"Precision: {precision:.4f} | Pred_Positive_Rate: {pred_positive_rate:.4f}"
        )

    results_df = pd.DataFrame(results)
    logger.info(f"\n=== Fairness Audit Results ===\n{results_df.to_string()}")
    return results_df


def plot_fairness(results_df, save_dir=None):
    if save_dir is None:
        save_dir = MODELS_DIR / "fairness"
    import os

    os.makedirs(save_dir, exist_ok=True)

    metrics = ["TPR (Recall)", "FPR", "Precision", "Pred_Positive_Rate"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axes[i].bar(results_df["group"], results_df[metric])
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)

    plt.suptitle("Fairness Audit by Gender")
    plt.tight_layout()
    plt.savefig(save_dir / "fairness_gender.png")
    plt.close()
    logger.info("Saved fairness plot")


def find_fair_threshold(y_test, y_proba, sensitive_col):
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        df = pd.DataFrame(
            {"y_true": y_test.values, "y_pred": y_pred, "group": sensitive_col.values}
        )

        tprs = {}
        for group, group_df in df.groupby("group"):
            if group == "XNA":
                continue
            tn, fp, fn, tp = confusion_matrix(
                group_df["y_true"], group_df["y_pred"], labels=[0, 1]
            ).ravel()
            tprs[group] = tp / (tp + fn) if (tp + fn) > 0 else 0

        if len(tprs) == 2:
            tpr_gap = abs(tprs["Male"] - tprs["Female"])
            avg_tpr = np.mean(list(tprs.values()))
            results.append(
                {
                    "threshold": round(threshold, 2),
                    "TPR_Male": round(tprs["Male"], 4),
                    "TPR_Female": round(tprs["Female"], 4),
                    "TPR_gap": round(tpr_gap, 4),
                    "avg_TPR": round(avg_tpr, 4),
                }
            )

    results_df = pd.DataFrame(results)

    # Find threshold with minimum TPR gap
    best = results_df.loc[results_df["TPR_gap"].idxmin()]
    logger.info("=== Fairest Threshold ===")
    logger.info(f"Threshold: {best['threshold']}")
    logger.info(f"TPR Male: {best['TPR_Male']}, TPR Female: {best['TPR_Female']}")
    logger.info(f"TPR Gap: {best['TPR_gap']}")
    logger.info(f"Avg TPR: {best['avg_TPR']}")

    return results_df, best


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

    # CODE_GENDER is encoded: 0=M, 1=F, 2=XNA
    gender_col = X_test["CODE_GENDER"].map({0: "Male", 1: "Female", 2: "XNA"})

    # Audit at both thresholds
    for threshold in [0.3, 0.5]:
        logger.info(f"\n=== Threshold: {threshold} ===")
        results_df = fairness_audit(y_test, ensemble_proba, gender_col, threshold)

    plot_fairness(results_df, save_dir=MODELS_DIR / "fairness")
    results_df, best_threshold = find_fair_threshold(y_test, ensemble_proba, gender_col)
