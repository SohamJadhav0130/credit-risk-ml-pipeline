import logging
import pandas as pd
from credit_pipeline.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.data = None

    def load_data(self, path: str):
        self.data = pd.read_csv(path)
        logger.info(
            f"Data loaded {self.data.shape[0]} rows and {self.data.shape[1]} columns"
        )
        return self

    def validate_data(self):
        if self.data is None:
            logger.error("No data loaded")
            return False
        if self.data.empty:
            logger.error("Data is empty")
            return False
        logger.info("Data validation passed")
        return True

    def display_data(self):
        logger.info(f"Data head:\n{self.data.head()}")
        return self

    def save_data(self):
        if self.validate_data():
            return self.data
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = DATA_DIR / "application_data.csv"

    loader = DataLoader()
    loader.load_data(path)
    loader.display_data()
    loader.save_data()
