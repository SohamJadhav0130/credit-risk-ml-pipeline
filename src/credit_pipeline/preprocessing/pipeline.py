import logging
logger = logging.getLogger(__name__)

from credit_pipeline.preprocessing.transformers import Transformers
from credit_pipeline.data.loader import DataLoader
from credit_pipeline.data.cleaner import DataCleaner
from credit_pipeline.features.engineering import FeaturesEngineering
from credit_pipeline.data.separator import DataSeparator
from credit_pipeline.data.splitter import DataSplitter
from credit_pipeline.data.merger import PreviousApplicationMerger
import joblib
from credit_pipeline.utils.config import load_config
    # 1. load config
    # 2. load data
    # merge prev data
    # 3. clean
    # 4. transform (fit + transform)
    # 5. feature engineering
     #6. split data (in splitter.py)
    # 7. separator
    # 8. impute missing values using correct domain knowledge(fit imputer on train, transform both train and test)
    # 9. save fitted transformer for later use in model training and inference
    # 9. return data AND fitted transformer

def build_pipeline(data_path, prev_path, config_path):
    config = load_config(config_path)
        
    threshold = config['missing_values']['threshold']
    excluded_attributes = config['missing_values']['excluded_attributes']
    flag_columns = config['flag_columns_correlation']['flag_columns']
    unrelated_items = config['unrelated_items_removal']['unrelated_items']
    correlation_threshold = config['flag_columns_correlation']['correlation_threshold']
    lower_bound = config['outliers_removal']['lower_bound']
    upper_bound = config['outliers_removal']['upper_bound']

    transformers = Transformers(
        threshold=threshold,
        excluded_attributes=excluded_attributes,
        flag_columns=flag_columns,
        unrelated_items=unrelated_items,
        correlation_threshold=correlation_threshold,
        outliers_removal_lower_bound=lower_bound,
        outliers_removal_upper_bound=upper_bound
    )
    
    #load data
    loader = DataLoader()
    loader.load_data(data_path)
    data = loader.save_data()
    if data is None:
        logger.error("No data loaded")
        exit(1)
    
    #merge previous application data
    merger = PreviousApplicationMerger()
    merger.load_and_aggregate(prev_path)
    data = merger.merge_with_main(data)
    #clean data
    cleaner = DataCleaner()
    data = cleaner.fit(data).perform_cleaning().get_data()

    #transform data
    transformers.fit(data)
    data = transformers.perform_cleaning().get_data()
    #feature engineering
    features_engineering = FeaturesEngineering(data,config_path)
    features_engineering.fit(data)
    data = features_engineering.transform(data)
    #splitter
    splitter = DataSplitter(config_path)
    X_train, X_test, y_train, y_test = splitter.split_data(data)
    
    #separator
    separator = DataSeparator()
    separator.fit(data)
    numeric_cols = separator.get_numeric_cols()
    categorical_cols = separator.get_categorical_cols()
    
    # impute missing values
    transformers.fit_imputer(X_train, numeric_cols)
    X_train = transformers.transform_imputer(X_train)
    X_test = transformers.transform_imputer(X_test)
    
    #PCA and clustering based features
    pca_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'TARGET']]
    features_engineering.fit_pca(X_train, pca_cols)
    X_train_pca = features_engineering.transform_pca(X_train, pca_cols)
    X_test_pca = features_engineering.transform_pca(X_test, pca_cols)
    
    features_engineering.fit_clusters(X_train_pca)
    X_train = features_engineering.transform_clusters(X_train_pca,X_train)
    X_test = features_engineering.transform_clusters(X_test_pca,X_test)
    
    # fit encoder on training data and transform both train and test
    transformers.fit_encoder(X_train, y_train, categorical_cols)
    X_train = transformers.transform_encoder(X_train)
    X_test = transformers.transform_encoder(X_test)
    
    joblib.dump(transformers, 'models/fitted_transformer.joblib')
    logger.info("Fitted transformer saved to models/fitted_transformer.joblib")
    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = "E:\\DCS Final Project\\credit-ml-pipeline\\data\\application_data.csv"
    config_path = "E:\\DCS Final Project\\credit-ml-pipeline\\config\\preprocessing_config.yaml"
    prev_path = "E:\\DCS Final Project\\credit-ml-pipeline\\data\\previous_application.csv"
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, transformers = build_pipeline(data_path, prev_path, config_path)

    logger.info(f"Final train data shape: {X_train.shape}")
    logger.info(f"Final test data shape: {X_test.shape}")
    logger.info(f"Missing values in X_train:\n{X_train.isnull().sum()[X_train.isnull().sum() > 0]}")
    logger.info(f"Missing values in X_test:\n{X_test.isnull().sum()[X_test.isnull().sum() > 0]}")     
    logger.info(f"Numerical cols:{numeric_cols}")
    logger.info(f"Categorical cols:{categorical_cols}")  

