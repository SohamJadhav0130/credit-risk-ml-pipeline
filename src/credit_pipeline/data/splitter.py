import logging
from credit_pipeline.utils.config import load_config
from sklearn.model_selection import train_test_split
from credit_pipeline.utils.paths import CONFIG_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO)


class DataSplitter:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.test_size = self.config["split_data"]["test_size"]
        self.random_state = self.config["split_data"]["random_state"]

    def split_data(self, data):
        X = data.drop(columns=["TARGET"])
        y = data["TARGET"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        logging.info(
            f"Data split into train and test sets with test size {self.test_size} and random state {self.random_state}"
        )
        logging.info(
            f"Train set shape: X_train: {X_train.shape}, y_train: {y_train.shape}"
        )
        logging.info(f"Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.info("Starting data splitting process")
    data_path = DATA_DIR / "application_data.csv"
    config_path = CONFIG_DIR / "preprocessing_config.yaml"

    # For testing purposes only, in actual pipeline this will be imported from pipeline.py
    # from credit_pipeline.preprocessing.pipeline import build_pipeline
    # splitter = DataSplitter(config_path)
    # data, numeric_cols, categorical_cols, transformers = build_pipeline(data_path, config_path)
    # X_train, X_test, y_train, y_test = splitter.split_data(data)
