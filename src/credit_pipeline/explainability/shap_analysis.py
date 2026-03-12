import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import XGBClassifier

from credit_pipeline.preprocessing.pipeline import build_pipeline
from credit_pipeline.utils.config import load_config
from credit_pipeline.utils.paths import CONFIG_DIR, DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)
SHAP_DIR = MODELS_DIR / "shap"


def global_importance(shap_values, X_sample):
    os.makedirs(SHAP_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample, plot_type="bar", max_display=20, show=False
    )
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_bar.png")
    plt.close()
    logger.info("Saved SHAP bar plot")

    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_beeswarm.png")
    plt.close()
    logger.info("Saved SHAP beeswarm plot")


def get_features_to_drop(shap_values, X_sample, threshold):
    mean_shap = np.abs(shap_values.values).mean(axis=0)  # add .values here
    feature_importance = dict(zip(X_sample.columns, mean_shap))
    to_drop = [f for f, v in feature_importance.items() if v < threshold]
    logger.info(f"Features to drop ({len(to_drop)}): {to_drop}")
    return to_drop


def individual_explanation(shap_values, y_sample):
    os.makedirs(SHAP_DIR, exist_ok=True)
    defaulter_idx = np.where(y_sample.values == 1)[0][0]
    non_defaulter_idx = np.where(y_sample.values == 0)[0][0]
    for label, idx in [
        ("defaulter", defaulter_idx),
        ("non_defaulter", non_defaulter_idx),
    ]:
        plt.figure()
        shap.plots.force(
            shap_values[idx], show=False
        )  # force plot instead of waterfall
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"force_{label}.png")
        plt.close()
        logger.info(f"Saved force plot for {label}")


def dependence_plots(shap_values, X_sample):
    os.makedirs(SHAP_DIR, exist_ok=True)
    top_features = ["EXT_SOURCE_3", "EXT_SOURCE_2", "SAVINGS_SCORE"]
    for feature in top_features:
        if feature in X_sample.columns:
            plt.figure()
            shap.dependence_plot(feature, shap_values.values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(SHAP_DIR / f"dependence_{feature}.png")
            plt.close()
            logger.info(f"Saved dependence plot for {feature}")


def feature_importance_report(shap_values, X_sample):
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = dict(zip(X_sample.columns, mean_shap))

    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )

    logger.info("\n=== Feature Importance Report ===")
    for rank, (feature, score) in enumerate(sorted_features, 1):
        logger.info(f"{rank:2d}. {feature:<40} {score:.4f}")

    return sorted_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_path = DATA_DIR / "application_data.csv"
    config_path = CONFIG_DIR / "preprocessing_config.yaml"
    prev_path = DATA_DIR / "previous_application.csv"

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers = (
        build_pipeline(data_path, prev_path, config_path)
    )

    config = load_config()
    drop_threshold = config["shap"]["drop_threshold"]
    sample_rows = config["shap"]["sample_rows"]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    xgb_best = XGBClassifier(
        n_estimators=775,
        max_depth=4,
        learning_rate=0.03281,
        subsample=0.8299,
        colsample_bytree=0.7808,
        min_child_weight=28,
        reg_alpha=0.4053,
        reg_lambda=0.9475,
        scale_pos_weight=neg / pos,
        random_state=42,
    )
    xgb_best.fit(X_train, y_train)

    sample_idx = np.random.choice(len(X_train), sample_rows, replace=False)
    X_sample = X_train.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_train.iloc[sample_idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(xgb_best)
    shap_values = explainer(X_sample)
    global_importance(shap_values, X_sample)  # Running Shap global prediction
    to_drop = get_features_to_drop(shap_values, X_sample, threshold=drop_threshold)
    individual_explanation(shap_values, y_sample)  # Individual prediction's explanation
    dependence_plots(shap_values, X_sample)  # Similar to correlation
    feature_importance_report(shap_values, X_sample)  # Numerical Report
