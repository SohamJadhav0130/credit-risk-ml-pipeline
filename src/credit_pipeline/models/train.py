import logging
from xgboost import XGBClassifier
import optuna
import joblib
# import mlflow

from credit_pipeline.utils.config import load_config
from lightgbm import LGBMClassifier as lgb
from catboost import CatBoostClassifier as cbc
from sklearn.metrics import average_precision_score
from credit_pipeline.preprocessing.pipeline import build_pipeline

logger = logging.getLogger(__name__)

# Train models:
# - XGBoost
# - LightGBM
# - CatBoost
# using MLFLow build a system that
# can be used to tracking experiements.


def model_training(X_train, X_test, y_train, y_test, scale_pos_weight):

    # XGBoost
    xgb_study = optuna.load_study(
        study_name="xgb_study", storage="sqlite:///logs/optuna/optuna_results.db"
    )
    xgb_best = XGBClassifier(
        **xgb_study.best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    xgb_best.fit(X_train, y_train)
    xgb_score = average_precision_score(y_test, xgb_best.predict_proba(X_test)[:, 1])
    logger.info(f"XGB AUC-PR on test set: {xgb_score}")
    joblib.dump(xgb_best, "models/xgb_best.joblib")

    # LightGBM
    lgb_study = optuna.load_study(
        study_name="lightGBM_study", storage="sqlite:///logs/optuna/optuna_results.db"
    )
    lgb_best = lgb(
        **lgb_study.best_params,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )
    lgb_best.fit(X_train, y_train)
    lgb_score = average_precision_score(y_test, lgb_best.predict_proba(X_test)[:, 1])
    logger.info(f"LightGBM AUC-PR on test set: {lgb_score}")
    joblib.dump(lgb_best, "models/lgb_best.joblib")

    # CatBoost
    catboost_study = optuna.load_study(
        study_name="catboost_study", storage="sqlite:///logs/optuna/optuna_results.db"
    )
    catboost_best = cbc(
        **catboost_study.best_params,
        class_weights=[1, scale_pos_weight],
        random_seed=42,
        verbose=0,
    )
    catboost_best.fit(X_train, y_train)
    catboost_score = average_precision_score(
        y_test, catboost_best.predict_proba(X_test)[:, 1]
    )
    logger.info(f"CatBoost AUC-PR on test set: {catboost_score}")
    joblib.dump(catboost_best, "models/catboost_best.joblib")

    return xgb_best, lgb_best, catboost_best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = "E:\\DCS Final Project\\credit-ml-pipeline\\data\\application_data.csv"
    config_path = (
        "E:\\DCS Final Project\\credit-ml-pipeline\\config\\preprocessing_config.yaml"
    )
    prev_path = (
        "E:\\DCS Final Project\\credit-ml-pipeline\\data\\previous_application.csv"
    )
    config = load_config(config_path)
    random_state = config["train"]["random_state"]

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers = (
        build_pipeline(data_path, prev_path, config_path)
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model_training(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scale_pos_weight=scale_pos_weight,
    )
