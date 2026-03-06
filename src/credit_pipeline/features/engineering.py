import logging

from credit_pipeline.utils.config import load_config
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeaturesEngineering:
    def __init__(self, data, config_path, X_train=None, X_test=None):
        self.data = data
        self.X_train = X_train
        self.X_test = X_test
        self.config_path = load_config(config_path)

        self.pca_variance_threshold = self.config_path["feature_engineering"][
            "pca_variance_threshold"
        ]
        self.random_state = self.config_path["feature_engineering"]["random_state"]
        self.n_clusters = self.config_path["feature_engineering"]["n_clusters_kmeans"]
        self.n_components_gmm = self.config_path["feature_engineering"][
            "n_components_gmm"
        ]

    def transform(self, data):
        curr_data = data.copy()
        # ANNUITY_TO_INCOME
        curr_data["ANNUITY_TO_INCOME"] = curr_data["AMT_ANNUITY"] / (
            curr_data["AMT_INCOME_TOTAL"] + 1
        )
        corr = curr_data["ANNUITY_TO_INCOME"].corr(curr_data["TARGET"])
        # SAVINGS_SCORE
        logger.info(
            f"ANNUITY_TO_INCOME feature created with min={curr_data['ANNUITY_TO_INCOME'].min()}, max={curr_data['ANNUITY_TO_INCOME'].max()}, mean={curr_data['ANNUITY_TO_INCOME'].mean()}, correlation with TARGET={corr}"
        )
        curr_data["SAVINGS_SCORE"] = (
            curr_data["EXT_SOURCE_2"] * curr_data["EXT_SOURCE_3"]
        )
        corr = curr_data["SAVINGS_SCORE"].corr(curr_data["TARGET"])
        # FAMILY_BURDEN_INDEX
        logger.info(
            f"SAVINGS_SCORE feature created with min={curr_data['SAVINGS_SCORE'].min()}, max={curr_data['SAVINGS_SCORE'].max()}, mean={curr_data['SAVINGS_SCORE'].mean()}, correlation with TARGET={corr}"
        )
        curr_data["FAMILY_BURDEN_INDEX"] = (
            curr_data["CNT_CHILDREN"] + curr_data["CNT_FAM_MEMBERS"]
        ) / (curr_data["AMT_INCOME_TOTAL"] + 1)
        corr = curr_data["FAMILY_BURDEN_INDEX"].corr(curr_data["TARGET"])
        logger.info(
            f"FAMILY_BURDEN_INDEX feature created with min={curr_data['FAMILY_BURDEN_INDEX'].min()}, max={curr_data['FAMILY_BURDEN_INDEX'].max()}, mean={curr_data['FAMILY_BURDEN_INDEX'].mean()}, correlation with TARGET={corr}"
        )
        # AGE_TO_EMPLOYMENT
        curr_data["AGE_TO_EMPLOYMENT"] = curr_data["AGE"] / (
            curr_data["YEARS_EMPLOYED"] + 1
        )
        corr = curr_data["AGE_TO_EMPLOYMENT"].corr(curr_data["TARGET"])
        logger.info(
            f"AGE_TO_EMPLOYMENT feature created with min={curr_data['AGE_TO_EMPLOYMENT'].min()}, max={curr_data['AGE_TO_EMPLOYMENT'].max()}, mean={curr_data['AGE_TO_EMPLOYMENT'].mean()}, correlation with TARGET={corr}"
        )
        # INCOME_PER_MEMBER
        curr_data["INCOME_PER_MEMBER"] = (
            curr_data["AMT_INCOME_TOTAL"] / curr_data["CNT_FAM_MEMBERS"]
        )
        corr = curr_data["INCOME_PER_MEMBER"].corr(curr_data["TARGET"])
        logger.info(
            f"INCOME_PER_MEMBER feature created with min={curr_data['INCOME_PER_MEMBER'].min()}, max={curr_data['INCOME_PER_MEMBER'].max()}, mean={curr_data['INCOME_PER_MEMBER'].mean()}, correlation with TARGET={corr}"
        )
        # CREDIT_PER_CHILD
        curr_data["CREDIT_PER_CHILD"] = curr_data["AMT_CREDIT"] / (
            curr_data["CNT_CHILDREN"] + 1
        )
        corr = curr_data["CREDIT_PER_CHILD"].corr(curr_data["TARGET"])
        logger.info(
            f"CREDIT_PER_CHILD feature created with min={curr_data['CREDIT_PER_CHILD'].min()}, max={curr_data['CREDIT_PER_CHILD'].max()}, mean={curr_data['CREDIT_PER_CHILD'].mean()}, correlation with TARGET={corr}"
        )
        # CREDIT_STRESS_RATIO
        curr_data["CREDIT_STRESS_RATIO"] = curr_data["AMT_CREDIT"] / (
            curr_data["AMT_INCOME_TOTAL"] + 1
        )
        corr = curr_data["CREDIT_STRESS_RATIO"].corr(curr_data["TARGET"])
        logger.info(
            f"CREDIT_STRESS_RATIO feature created with min={curr_data['CREDIT_STRESS_RATIO'].min()}, max={curr_data['CREDIT_STRESS_RATIO'].max()}, mean={curr_data['CREDIT_STRESS_RATIO'].mean()}, correlation with TARGET={corr}"
        )
        # CREDITS_TO_GOOD_RATIO
        curr_data["CREDITS_TO_GOOD_RATIO"] = curr_data["AMT_CREDIT"] / (
            curr_data["AMT_GOODS_PRICE"] + 1
        )
        corr = curr_data["CREDITS_TO_GOOD_RATIO"].corr(curr_data["TARGET"])
        logger.info(
            f"CREDITS_TO_GOOD_RATIO feature created with min={curr_data['CREDITS_TO_GOOD_RATIO'].min()}, max={curr_data['CREDITS_TO_GOOD_RATIO'].max()}, mean={curr_data['CREDITS_TO_GOOD_RATIO'].mean()}, correlation with TARGET={corr}"
        )
        # EXT_SOURCE_COMPOSITE
        curr_data["EXT_SOURCE_COMPOSITE"] = curr_data[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].mean(axis=1)
        corr = curr_data["EXT_SOURCE_COMPOSITE"].corr(curr_data["TARGET"])
        logger.info(
            f"EXT_SOURCE_COMPOSITE feature created with min={curr_data['EXT_SOURCE_COMPOSITE'].min()}, max={curr_data['EXT_SOURCE_COMPOSITE'].max()}, mean={curr_data['EXT_SOURCE_COMPOSITE'].mean()}, correlation with TARGET={corr}"
        )

        self.data = curr_data
        return curr_data

    def fit(self, data):
        self.data = data
        return self

    def print_features_info(self):
        logger.info(f"Features info: {self.data.columns.tolist()}")
        return self.data.columns.tolist()

    def fit_pca(self, X_train=None, numeric_cols=None):
        X_train = X_train.copy()
        # PCA on train
        X_train_numeric = X_train[numeric_cols]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_numeric)
        pca = PCA(n_components=self.pca_variance_threshold)
        pca.fit(X_train_scaled)
        self.pca = pca
        self.scaler = scaler
        logger.info(
            f"Fitted PCA with {pca.n_components_} components to retain {self.pca_variance_threshold * 100}% variance"
        )
        return self

    def transform_pca(self, X, numeric_cols=None):
        X = X.copy()
        X_numeric = X[numeric_cols]
        X_scaled = self.scaler.transform(X_numeric)  # PCA on fitted objects
        X_transformed = self.pca.transform(X_scaled)
        return X_transformed

    def fit_clusters(self, X_pca):
        # K-means with optimal k = 3 found from elbow method in EDA
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(X_pca)
        self.kmeans = kmeans
        logger.info(f"Fitted KMeans with {self.n_clusters} clusters")

        # GMM with optimal n = 6 found from elbow plot in EDA
        gmm = GaussianMixture(
            n_components=self.n_components_gmm, random_state=self.random_state
        )
        gmm.fit(X_pca)
        self.gmm = gmm
        logger.info(f"Fitted GaussianMixture with {self.n_components_gmm} components")

        return self

    def transform_clusters(self, X_pca, X_original):
        X_original = X_original.copy()
        X_original["KMeans_Clusters"] = self.kmeans.predict(X_pca)
        X_original["GMM_Clusters"] = self.gmm.predict(X_pca)
        logger.info(
            "Added KMeans_Clusters and GMM_Clusters features based on fitted cluster models"
        )
        return X_original


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = "E:\\DCS Final Project\\credit-ml-pipeline\\data\\application_data.csv"
    config_path = (
        "E:\\DCS Final Project\\credit-ml-pipeline\\config\\preprocessing_config.yaml"
    )
    # loader = DataLoader()
    # loader.load_data(path)
    # data = loader.save_data()
    # if data is None:
    #     logger.error("No data loaded")
    #     exit(1)
    # cleaner = DataCleaner()
    # data = cleaner.fit(data).perform_cleaning().get_data()

    # threshold = config_path['missing_values']['threshold']
    # excluded_attributes = config_path['missing_values']['excluded_attributes']
    # flag_columns = config_path['flag_columns_correlation']['flag_columns']
    # unrelated_items = config_path['unrelated_items_removal']['unrelated_items']
    # correlation_threshold = config_path['flag_columns_correlation']['correlation_threshold']
    # lower_bound = config_path['outliers_removal']['lower_bound']
    # upper_bound = config_path['outliers_removal']['upper_bound']

    # transformers = Transformers(
    #     threshold=threshold,
    #     excluded_attributes=excluded_attributes,
    #     flag_columns=flag_columns,
    #     unrelated_items=unrelated_items,
    #     correlation_threshold=correlation_threshold,
    #     outliers_removal_lower_bound=lower_bound,
    #     outliers_removal_upper_bound=upper_bound
    # )
    # transformers.fit(data)
    # transformed_data = transformers.perform_cleaning().get_data()

    # features_engineering = FeaturesEngineering(transformed_data)
    # features_engineering.fit(transformed_data)
    # transformed_data = features_engineering.transform(transformed_data)
    # features_engineering.print_features_info()
