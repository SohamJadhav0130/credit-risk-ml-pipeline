import logging
logger = logging.getLogger(__name__)

from sklearn.impute import SimpleImputer
from credit_pipeline.preprocessing.pipeline import PipelineSteps
import pandas as pd
import numpy as np

class Transformers(PipelineSteps):
    def __init__(self, threshold: float, excluded_attributes: list, flag_columns: list, unrelated_items: list, correlation_threshold: float, outliers_removal_lower_bound: float, outliers_removal_upper_bound: float):
        self.data = None
        self.threshold = threshold
        self.excluded_attributes = excluded_attributes
        self.flag_columns = flag_columns
        self.unrelated_items = unrelated_items
        self.correlation_threshold = correlation_threshold
        self.outliers_removal_lower_bound = outliers_removal_lower_bound
        self.outliers_removal_upper_bound = outliers_removal_upper_bound
    def fit(self, data):
        self.data = data.copy()
        return self
    def get_data(self):
        return self.data

    def missing_values_removal(self, X):
        # Missing values removal
        X = X.copy()
        missing_percentages = X.isnull().mean() * 100
        high_missing_cols = missing_percentages[missing_percentages > self.threshold].index.tolist()
        high_missing_cols = [col for col in high_missing_cols if col not in self.excluded_attributes]
        X = X.drop(columns=high_missing_cols)
        logger.info(f"Removed {len(high_missing_cols)} columns with missing values greater than {self.threshold}%")
        return X

    def unrelated_items_removal(self, X):
        X = X.copy()
        X = X.drop(columns=[col for col in self.unrelated_items if col in X.columns])
        logger.info(f"Removed {len(self.unrelated_items)} unrelated items")
        return X
    
    def flag_columns_correlation(self, X):
        # Pearson correlation
        X = X.copy()
        flag_cols = [col for col in self.flag_columns if col in X.columns]
        for flag in flag_cols:
            # Only check if both columns are present
            if flag in X.columns and "TARGET" in X.columns:
                try:
                    correlation = X[[flag, 'TARGET']].corr().iloc[0, 1]
                    if pd.isna(correlation) or abs(correlation) < self.correlation_threshold:
                        X = X.drop(columns=flag)
                        logger.info(f"Removed {flag} column with correlation less than {self.correlation_threshold}")
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for {flag}: {e}")
        return X

    def outliers_removal(self, X):
        X = X.copy()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Columns that should not be capped
        skip_cols = ['TARGET', 'SK_ID_CURR', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY']
        num_cols = [col for col in num_cols if col not in skip_cols]
        for col in num_cols:
            original = X[col]
            lower_bound = original.quantile(self.outliers_removal_lower_bound)
            upper_bound = original.quantile(self.outliers_removal_upper_bound)
            X[col] = np.clip(original, lower_bound, upper_bound)
            num_outliers = ((original < lower_bound) | (original > upper_bound)).sum()
            logger.info(f"Column '{col}': {num_outliers} outliers capped at [{lower_bound:.2f}, {upper_bound:.2f}]")
        return X

    # def impute_numeric_cols(self, X):
    #     #SimpleImputer module to handle the numeric cols missing values
    #     X = X.copy()
    #     num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    #     X[num_cols] = self.imputer.transform(X[num_cols])
    #     logger.info(f"Imputed {len(num_cols)} numeric columns with median value")
    #     logger.info(f"Unique values in numeric columns after imputation: {X[num_cols].nunique()}")
    #     return X
    def duplicate_rows_and_inf_values(self, X):
        X = X.copy()
        duplicate_rows = X.duplicated().sum()
        logger.info(f"Duplicate rows: {duplicate_rows}")
        X = X.drop_duplicates()
        logger.info(f"Removed {duplicate_rows} duplicate rows")
        inf_values = X.isin([np.inf, -np.inf]).values.sum()
        logger.info(f"Inf values: {inf_values}")
        X = X.replace([np.inf, -np.inf], np.nan)
        logger.info(f"Replaced {inf_values} inf values with NaN")
        return X

    def perform_cleaning(self):
        self.data = self.missing_values_removal(self.data)
        self.data = self.unrelated_items_removal(self.data)
        self.data = self.flag_columns_correlation(self.data)
        self.data = self.duplicate_rows_and_inf_values(self.data)
        self.data = self.outliers_removal(self.data)
        #self.data = self.data.fillna(self.data.median())
        logger.info("Data cleaning completed")
        logger.info(f"Data shape after transformation: {self.data.shape}")
        logger.info(f"Data head after transformation: {self.data.head()}")
        logger.info(f"Data missing values after transformation:\n {(self.data.isnull().mean() * 100).sort_values(ascending=False)}")
        logger.info("Transformation completed")
        return self

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from credit_pipeline.data.loader import DataLoader
    from credit_pipeline.data.cleaner import DataCleaner
    import yaml
    path = "E:\\DCS Final Project\\credit-ml-pipeline\\data\\application_data.csv"
    application_data = DataLoader()
    application_data.load_data(path)
    if application_data.save_data() is None:
        logger.error("No data loaded")
        exit(1)
    cleaner = DataCleaner()
    data = cleaner.fit(application_data.save_data()).perform_cleaning().get_data()
    if data is None:
        logger.error("No data cleaned")
        exit(1)
    
    with open('E:\\DCS Final Project\\credit-ml-pipeline\\config\\preprocessing_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    threshold = config['missing_values']['threshold']
    excluded_attributes = config['missing_values']['excluded_attributes']
    flag_columns = config['flag_columns_correlation']['flag_columns']
    unrelated_items = config['unrelated_items_removal']['unrelated_items']
    correlation_threshold = config['flag_columns_correlation']['correlation_threshold']
    lower_bound = config['outliers_removal']['lower_bound']
    upper_bound = config['outliers_removal']['upper_bound']
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Excluded attributes: {excluded_attributes}")
    logger.info(f"Flag columns: {flag_columns}")
    logger.info(f"Unrelated items: {unrelated_items}")
    logger.info(f"Correlation threshold: {correlation_threshold}")
    logger.info(f"Lower bound: {lower_bound}")
    logger.info(f"Upper bound: {upper_bound}")

    transformers = Transformers(
        threshold=threshold,
        excluded_attributes=excluded_attributes,
        flag_columns=flag_columns,
        unrelated_items=unrelated_items,
        correlation_threshold=correlation_threshold,
        outliers_removal_lower_bound=lower_bound,
        outliers_removal_upper_bound=upper_bound
    )
    transformers.fit(data)
    transformed_data = transformers.perform_cleaning().get_data()