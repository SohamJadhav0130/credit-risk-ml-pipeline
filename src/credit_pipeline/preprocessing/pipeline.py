import logging
import os
import pandas as pd
import joblib

from credit_pipeline.data.cleaner import DataCleaner
from credit_pipeline.data.loader import DataLoader
from credit_pipeline.data.merger import PreviousApplicationMerger
from credit_pipeline.data.separator import DataSeparator
from credit_pipeline.data.splitter import DataSplitter
from credit_pipeline.features.engineering import FeaturesEngineering
from credit_pipeline.preprocessing.transformers import Transformers
from credit_pipeline.utils.config import load_config
from credit_pipeline.utils.paths import CONFIG_DIR, DATA_DIR, CACHE_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

# 1. load config
# 2. load data3
# merge prev data
# 3. clean
# 4. transform (fit + transform)
# 5. feature engineering
# 6. split data (in splitter.py)
# 7. separator
# 8. impute missing values using correct domain knowledge(fit imputer on train, transform both train and test)
# 9. save fitted transformer for later use in model training and inference
# 9. return data AND fitted transformer


def build_pipeline(data_path, prev_path, config_path, model_path=None):
    config = load_config(config_path)

    threshold = config["missing_values"]["threshold"]
    excluded_attributes = config["missing_values"]["excluded_attributes"]
    flag_columns = config["flag_columns_correlation"]["flag_columns"]
    unrelated_items = config["unrelated_items_removal"]["unrelated_items"]
    correlation_threshold = config["flag_columns_correlation"]["correlation_threshold"]
    lower_bound = config["outliers_removal"]["lower_bound"]
    upper_bound = config["outliers_removal"]["upper_bound"]

    transformers = Transformers(
        threshold=threshold,
        excluded_attributes=excluded_attributes,
        flag_columns=flag_columns,
        unrelated_items=unrelated_items,
        correlation_threshold=correlation_threshold,
        outliers_removal_lower_bound=lower_bound,
        outliers_removal_upper_bound=upper_bound,
    )

    # load data
    loader = DataLoader()
    loader.load_data(data_path)
    data = loader.save_data()
    if data is None:
        logger.error("No data loaded")
        exit(1)
    os.makedirs(CACHE_DIR, exist_ok=True)

    MERGED_CACHE = CACHE_DIR / "merged_data.parquet"
    CLEANED_CACHE = CACHE_DIR / "cleaned_data.parquet"
    # MERGE - check cache first
    if MERGED_CACHE.exists():
        logger.info("Loading merged data from cache")
        data = pd.read_parquet(MERGED_CACHE)
    else:
        merger = PreviousApplicationMerger()
        merger.load_and_aggregate(prev_path)
        data = merger.merge_with_main(data)
        data.to_parquet(MERGED_CACHE)
        logger.info("Saved merged data to cache")

    # CLEAN - check cache first
    if CLEANED_CACHE.exists():
        logger.info("Loading cleaned data from cache")
        data = pd.read_parquet(CLEANED_CACHE)
    else:
        cleaner = DataCleaner()
        data = cleaner.fit(data).perform_cleaning().get_data()
        data.to_parquet(CLEANED_CACHE)
        logger.info("Saved cleaned data to cache")

    # transform data
    transformers.fit(data)
    data = transformers.perform_cleaning().get_data()
    # feature engineering
    features_engineering = FeaturesEngineering(data, config_path)
    features_engineering.fit(data)
    data = features_engineering.transform(data)
    # splitter
    splitter = DataSplitter(config_path)
    X_train, X_test, y_train, y_test = splitter.split_data(data)

    # separator
    separator = DataSeparator()
    separator.fit(data)
    numeric_cols = separator.get_numeric_cols()
    categorical_cols = separator.get_categorical_cols()

    # impute missing values
    transformers.fit_imputer(X_train, numeric_cols)
    X_train = transformers.transform_imputer(X_train)
    X_test = transformers.transform_imputer(X_test)

    # PCA and clustering based features
    pca_cols = [col for col in numeric_cols if col not in ["SK_ID_CURR", "TARGET"]]
    features_engineering.fit_pca(X_train, pca_cols)
    X_train_pca = features_engineering.transform_pca(X_train, pca_cols)
    X_test_pca = features_engineering.transform_pca(X_test, pca_cols)

    features_engineering.fit_clusters(X_train_pca)
    X_train = features_engineering.transform_clusters(X_train_pca, X_train)
    X_test = features_engineering.transform_clusters(X_test_pca, X_test)

    # fit encoder on training data and transform both train and test
    transformers.fit_encoder(X_train, y_train, categorical_cols)
    X_train = transformers.transform_encoder(X_train)
    X_test = transformers.transform_encoder(X_test)

    # Did Optuna,ML model training and MLFlow for tracking and then SHAP.
    # retraining after shap analysis with low important features to be dropped.

    shap_drop = config.get("shap", {}).get("features_to_drop", [])
    existing_drop = [f for f in shap_drop if f in X_train.columns]
    X_train = X_train.drop(columns=existing_drop)
    X_test = X_test.drop(columns=existing_drop)
    logger.info(
        f"Dropped {len(existing_drop)} low-importance features: {existing_drop}"
    )
    if model_path is None:
        model_path = MODELS_DIR / "fitted_transformer.joblib"
    os.makedirs(model_path.parent, exist_ok=True)

    joblib.dump(transformers, model_path)
    logger.info(f"Fitted transformer saved to {model_path}")
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_cols,
        categorical_cols,
        transformers,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = DATA_DIR / "application_data.csv"
    config_path = CONFIG_DIR / "preprocessing_config.yaml"
    prev_path = DATA_DIR / "previous_application.csv"
    model_path = MODELS_DIR / "fitted_transformer.joblib"

    print(f"DEBUG model_path: {model_path}")  # add this
    print(f"DEBUG MODELS_DIR: {MODELS_DIR}")  # add this
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers = (
        build_pipeline(data_path, prev_path, config_path, model_path)
    )

    logger.info(f"Final train data shape: {X_train.shape}")
    logger.info(f"Final test data shape: {X_test.shape}")
    logger.info(
        f"Missing values in X_train:\n{X_train.isnull().sum()[X_train.isnull().sum() > 0]}"
    )
    logger.info(
        f"Missing values in X_test:\n{X_test.isnull().sum()[X_test.isnull().sum() > 0]}"
    )
    logger.info(f"Numerical cols:{numeric_cols}")
    logger.info(f"Categorical cols:{categorical_cols}")
