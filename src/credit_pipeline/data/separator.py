import logging

logger = logging.getLogger(__name__)


class DataSeparator:
    def __init__(self):
        self.numeric_cols = None
        self.categorical_cols = None

    def fit(self, data):
        self.numeric_cols = data.select_dtypes(
            include=["int32", "int64", "float64"]
        ).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
        return self

    def get_numeric_cols(self):
        logger.info(f"Numeric columns: {self.numeric_cols}")
        return self.numeric_cols

    def get_categorical_cols(self):
        logger.info(f"Categorical columns: {self.categorical_cols}")
        return self.categorical_cols


# from credit_pipeline.utils.config import load_config
# from credit_pipeline.preprocessing.pipeline import build_pipeline
# For individual testing purposes only
# if __name__ == "__main__":

#     logging.basicConfig(level=logging.INFO)
#     data_path = "E:\\DCS Final Project\\credit-ml-pipeline\\data\\application_data.csv"
#     config_path = 'E:\\DCS Final Project\\credit-ml-pipeline\\config\\preprocessing_config.yaml'
#     data, numeric_cols, categorical_cols, transformers = build_pipeline(data_path, config_path)

#     separator = DataSeparator()
#     separator.fit(data)
#     numeric_cols = separator.get_numeric_cols()
#     categorical_cols = separator.get_categorical_cols()
