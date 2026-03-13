# import mlflow
import logging

import optuna
from catboost import CatBoostClassifier as cbc
from credit_pipeline.preprocessing.pipeline import build_pipeline
from credit_pipeline.utils.config import load_config
from lightgbm import LGBMClassifier as lgb
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier as xgb
import numpy as np
from credit_pipeline.utils.paths import CONFIG_DIR, DATA_DIR, OPTUNA_DB_PATH

# optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)
# mlflow.set_experiment("First Trial")
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")


def xgb_objective(trial, X_train, y_train, neg, pos, cv_folds, random_state):
    # Hyperparameters for XGB
    n_estimators = trial.suggest_int("n_estimators", 300, 800)
    max_depth = trial.suggest_int("max_depth", 4, 7)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    subsample = trial.suggest_float("subsample", 0.7, 0.9)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.7, 0.9)
    min_child_weight = trial.suggest_int("min_child_weight", 5, 30)
    reg_alpha = trial.suggest_float("reg_alpha", 0.01, 1.0)  # add L1
    reg_lambda = trial.suggest_float("reg_lambda", 0.01, 1.0)  # add L2
    scale_pos_weight = neg / pos  # fixed, not tuned

    # Create XGB with suggested hypeparameters
    xgb_model = xgb(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    score = cross_val_score(
        xgb_model, X_train, y_train, cv=cv, scoring="average_precision"
    ).mean()
    return score


def catboost_objective(trial, X_train, y_train, neg, pos, cv_folds, random_state):
    iterations = trial.suggest_int("iterations", 300, 800)
    depth = trial.suggest_int("depth", 4, 7)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    border_count = trial.suggest_int("border_count", 32, 255)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        cbc_model = cbc(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            border_count=border_count,
            class_weights=[1, neg / pos],
            random_seed=random_state,
            verbose=0,
        )
        cbc_model.fit(X_tr, y_tr)
        y_pred = cbc_model.predict_proba(X_val)[:, 1]

        from sklearn.metrics import average_precision_score

        scores.append(average_precision_score(y_val, y_pred))

    score = np.mean(scores)
    return score


def lightGBM_objective(trial, X_train, y_train, neg, pos, cv_folds, random_state):
    # Hyperparameters for LightGBM
    n_estimators = trial.suggest_int("n_estimators", 300, 800)
    max_depth = trial.suggest_int("max_depth", 4, 7)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    num_leaves = trial.suggest_int("num_leaves", 31, 100)
    min_child_samples = trial.suggest_int("min_child_samples", 20, 100)
    subsample = trial.suggest_float("subsample", 0.7, 0.9)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.7, 0.9)
    reg_alpha = trial.suggest_float("reg_alpha", 0.01, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.01, 1.0)
    is_unbalance = True  # fixed

    lightGBM_model = lgb(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        is_unbalance=is_unbalance,
        random_state=random_state,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    score = cross_val_score(
        lightGBM_model, X_train, y_train, cv=cv, scoring="average_precision"
    ).mean()
    return score


def rf_objective(trial, X_train, y_train, neg, pos, cv_folds, random_state):
    # Hyperparameters for RF
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    class_weight = "balanced"  # fixed

    rf_model = rf(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    score = cross_val_score(
        rf_model, X_train, y_train, cv=cv, scoring="average_precision"
    ).mean()
    return score


# Log Model
def fit_logistic_regression(X_train, y_train, cv_folds, random_state):
    lr_model = lr(C=0.1, class_weight="balanced", max_iter=1000, solver="saga")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    score = cross_val_score(
        lr_model, X_train, y_train, cv=cv, scoring="average_precision"
    ).mean()
    logger.info(f"LogReg score: {score}")
    return lr_model, score


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Started the Optuna Tuning")

    data_path = DATA_DIR / "application_data.csv"
    config_path = CONFIG_DIR / "preprocessing_config.yaml"
    prev_path = DATA_DIR / "previous_application.csv"

    config = load_config(config_path)
    random_state = config["optuna_tuning"]["random_state"]
    cv_folds = config["optuna_tuning"]["cv_folds"]
    n_trials = config["optuna_tuning"]["n_trials"]

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers = (
        build_pipeline(data_path, prev_path, config_path)
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    """xgb_obj = partial(
        xgb_objective,
        X_train=X_train,
        y_train=y_train,
        neg=neg,
        pos=pos,
        cv_folds=cv_folds,
        random_state=random_state,
    )"""
    # Lambda shrinks the above defined function into one single line
    # Optuna calls this internally like: objective(trial)
    # Lambda intercepts and adds the extra args
    # sampler=optuna.samplers.TPESampler() --> Bayesian Sampler

    xgb_study = optuna.create_study(
        direction="maximize",
        study_name="xgb_study",
        storage=f"sqlite:///{OPTUNA_DB_PATH}",
        load_if_exists=True,
    )
    xgb_study.optimize(
        lambda trial: xgb_objective(
            trial, X_train, y_train, neg, pos, cv_folds, random_state
        ),
        n_trials=n_trials,
        n_jobs=3,
    )

    catboost_study = optuna.create_study(
        direction="maximize",
        study_name="catboost_study",
        storage=f"sqlite:///{OPTUNA_DB_PATH}",
        load_if_exists=True,
    )
    catboost_study.optimize(
        lambda trial: catboost_objective(
            trial, X_train, y_train, neg, pos, cv_folds, random_state
        ),
        n_trials=n_trials,
        n_jobs=3,
    )

    lightGBM_study = optuna.create_study(
        direction="maximize",
        study_name="lightGBM_study",
        storage=f"sqlite:///{OPTUNA_DB_PATH}",
        load_if_exists=True,
    )
    lightGBM_study.optimize(
        lambda trial: lightGBM_objective(
            trial, X_train, y_train, neg, pos, cv_folds, random_state
        ),
        n_trials=n_trials,
        n_jobs=3,
    )

    # # Log Reg fitted only with one trial
    fit_logistic_regression(X_train, y_train, cv_folds, random_state)

    # # RF
    random_forest_study = optuna.create_study(
        direction="maximize",
        study_name="rf_study",
        storage=f"sqlite:///{OPTUNA_DB_PATH}",
        load_if_exists=True,
    )
    random_forest_study.optimize(
        lambda trial: rf_objective(
            trial, X_train, y_train, neg, pos, cv_folds, random_state
        ),
        n_trials=n_trials,
        n_jobs=3,
    )

    # log best results and save models
    logger.info(
        f"XGB best score: {xgb_study.best_value}, params: {xgb_study.best_params}"
    )
    logger.info(
        f"CatBoost best score: {catboost_study.best_value}, params: {catboost_study.best_params}"
    )
    logger.info(
        f"LightGBM best score: {lightGBM_study.best_value}, params: {lightGBM_study.best_params}"
    )
    logger.info(
        f"RF best score: {random_forest_study.best_value}, params: {random_forest_study.best_params}"
    )
