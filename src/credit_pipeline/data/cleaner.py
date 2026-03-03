import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.data = None

    def fit(self, data):
        self.data = data.copy()
        return self

    def fix_family_size(self):
        """
        Validate and clean family structure variables
        Rules:
        1. Remove negative children (impossible)
        2. Remove children > 14 (unrealistic for loan applicants)
        3. Ensure family_members >= children + 1 (logical constraint)
        4. Cap family_members at 95th percentile (handle outliers softly)
        """
        before = len(self.data)
        self.data = self.data[self.data['CNT_CHILDREN'] >= 0]
        self.data = self.data[self.data['CNT_CHILDREN'] <= 14]
        removed = before - len(self.data)
        logger.info(f"Removed {removed} rows with invalid children count")

        before = len(self.data)
        self.data = self.data[self.data['CNT_FAM_MEMBERS'] >= self.data['CNT_CHILDREN'] + 1]
        removed = before - len(self.data)
        logger.info(f"Removed {removed} rows where family members < children + 1")

        before_cap = self.data['CNT_FAM_MEMBERS'].max()
        fam_cap = self.data['CNT_FAM_MEMBERS'].quantile(0.95)
        self.data['CNT_FAM_MEMBERS'] = np.where(
            self.data['CNT_FAM_MEMBERS'] > fam_cap,
            fam_cap,
            self.data['CNT_FAM_MEMBERS']
        )
        capped_count = (self.data['CNT_FAM_MEMBERS'] == fam_cap).sum()
        logger.info(f"Capped {capped_count} rows with family members at {fam_cap} (95th percentile), max before cap was {before_cap}")
        return self

    def replace_org_name(self):
        xna_cols = (self.data['ORGANIZATION_TYPE'] == 'XNA').sum()
        logger.info(f"Replacing {xna_cols} 'XNA' values in ORGANIZATION_TYPE with 'Undefined org. type'")
        self.data['ORGANIZATION_TYPE'] = self.data['ORGANIZATION_TYPE'].replace('XNA', 'Undefined org. type')
        return self

    def transform_days(self):
        """
        Transform DAYS_* columns to interpretable features:
        - AGE: from DAYS_BIRTH
        - YEARS_EMPLOYED: from DAYS_EMPLOYED
        - YEARS_ID_PUBLISH: from DAYS_ID_PUBLISH
        """
        self.data['AGE'] = (-self.data['DAYS_BIRTH'] / 365).astype(int)
        self.data['YEARS_EMPLOYED'] = (-self.data['DAYS_EMPLOYED'] / 365).astype(int)
        self.data['YEARS_ID_PUBLISH'] = (-self.data['DAYS_ID_PUBLISH'] / 365).astype(int)
        logger.info("Transformed DAYS_BIRTH to AGE, DAYS_EMPLOYED to YEARS_EMPLOYED, and DAYS_ID_PUBLISH to YEARS_ID_PUBLISH")

        org_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH']
        self.data.drop(org_cols, axis=1, inplace=True)
        logger.info(f"Dropped original DAYS_* columns: {org_cols}")
        return self

    def cap_income(self):
        before = len(self.data)
        self.data = self.data[self.data['AMT_INCOME_TOTAL'] > 0]
        income_cap = self.data['AMT_INCOME_TOTAL'].quantile(0.99)
        self.data['AMT_INCOME_TOTAL'] = np.clip(self.data['AMT_INCOME_TOTAL'], None, income_cap)
        logger.info(f"Removed {before - len(self.data)} rows with impossible income values")
        logger.info(f"Capped income values at 99th percentile: {income_cap}")
        return self

    def impute_years_employed(self):
        missing_employment = (self.data['YEARS_EMPLOYED'] == -1000).sum()
        logger.info(f"Identified {missing_employment} rows with -1000 in YEARS_EMPLOYED")
        self.data['YEARS_EMPLOYED'] = self.data['YEARS_EMPLOYED'].replace(-1000, np.nan)
        median_years = self.data['YEARS_EMPLOYED'].median()
        self.data['YEARS_EMPLOYED'] = self.data['YEARS_EMPLOYED'].fillna(median_years).astype(int)
        logger.info(f"Imputed missing values in YEARS_EMPLOYED with median value: {median_years}")
        logger.info(f"Unique values in YEARS_EMPLOYED after imputation: {self.data['YEARS_EMPLOYED'].unique()}")
        return self

    def clean_occupation_type(self):
        self.data['OCCUPATION_TYPE'] = self.data['OCCUPATION_TYPE'].fillna('Unknown')
        self.data['OCCUPATION_TYPE'] = self.data['OCCUPATION_TYPE'].replace('Unknown', 'Undefined occupation')
        logger.info(f"Cleaned OCCUPATION_TYPE: {self.data['OCCUPATION_TYPE'].unique()}")
        return self

    def print_missing_percentage(self):
        logger.info(f"Missing percentage:\n{(self.data.isnull().mean() * 100).sort_values(ascending=False)}")
        return self

    def perform_cleaning(self):
        self.fix_family_size()
        self.replace_org_name()
        self.transform_days()
        self.cap_income()
        self.impute_years_employed()
        self.clean_occupation_type()
        self.print_missing_percentage()
        logger.info("Data cleaning completed")
        return self

    def get_data(self):
        return self.data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from loader import DataLoader
    path = "E:\\DCS Final Project\\credit-ml-pipeline\\data\\application_data.csv"
    application_data = DataLoader()
    application_data.load_data(path)
    raw_data = application_data.save_data()
    if raw_data is None:
        logger.error("No data loaded")
        exit(1)
    cleaner = DataCleaner()
    cleaned_data = cleaner.fit(raw_data).perform_cleaning().get_data()
    logger.info(f"Cleaned data has {len(cleaned_data)} rows and {len(cleaned_data.columns)} columns")
    logger.info(f"Head of cleaned data:\n{cleaned_data.head()}")