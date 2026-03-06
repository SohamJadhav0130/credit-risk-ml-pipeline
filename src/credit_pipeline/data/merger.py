# This file merges the second dataset "previous_application.csv" with the main dataset "application_data.csv" as a left join
# based on the common key "SK_ID_CURR". The merged dataset is then saved as "merged_data.csv" for further processing in the pipeline.

"""
Merged only behavioural columns:-
Approval behaviour:
- Total previous applications
- Number approved, refused, cancelled
- Approval rate

Financial behaviour:
- Mean/max previous credit amount
- Mean previous annuity
- Credit vs application amount ratio (did they get less than they asked?)

Loan characteristics:
- Mean payment count (loan tenure)
- Mean days since decision
- Most recent contract type"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PreviousApplicationMerger:
    def __init__(self):
        self.data = None

    def load_and_aggregate(self, prev_path):
        prev_data = pd.read_csv(prev_path)
        logger.info(f"Previous application data loaded with shape: {prev_data.shape}")

        prev_data["NAME_CONTRACT_STATUS"] = prev_data["NAME_CONTRACT_STATUS"].replace(
            "XNA", np.nan
        )
        prev_data["CREDIT_TO_APPLICATION"] = prev_data["AMT_CREDIT"] / (
            prev_data["AMT_APPLICATION"] + 1
        )

        self.agg_data = (
            prev_data.groupby("SK_ID_CURR")
            .agg(
                Prev_Count=("SK_ID_PREV", "count"),
                Prev_Approved=(
                    "NAME_CONTRACT_STATUS",
                    lambda x: (x == "Approved").sum(),
                ),
                Prev_Refused=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
                Prev_Cancelled=(
                    "NAME_CONTRACT_STATUS",
                    lambda x: (x == "Canceled").sum(),
                ),
                Prev_Approval_Rate=(
                    "NAME_CONTRACT_STATUS",
                    lambda x: (x == "Approved").mean(),
                ),
                Prev_Mean_Credit=("AMT_CREDIT", "mean"),
                Prev_Max_Credit=("AMT_CREDIT", "max"),
                Prev_Mean_Annuity=("AMT_ANNUITY", "mean"),
                Prev_Application_Mean=("AMT_APPLICATION", "mean"),
                Prev_Credit_Application_Ratio=("CREDIT_TO_APPLICATION", "mean"),
                Prev_Mean_Payment_Count=("CNT_PAYMENT", "mean"),
                Prev_Mean_Days_Decision=("DAYS_DECISION", "mean"),
            )
            .reset_index()
        )

        logger.info(
            f"Aggregated previous application data with shape: {self.agg_data.shape}"
        )
        return self

    def merge_with_main(self, main_data):
        merged = main_data.merge(self.agg_data, on="SK_ID_CURR", how="left")

        # applications without previous applications will get 0 for counts.
        count_cols = ["Prev_Count", "Prev_Approved", "Prev_Refused", "Prev_Cancelled"]
        merged[count_cols] = merged[count_cols].fillna(0)

        # fill rate and amount columns with median
        amount_cols = [
            col
            for col in self.agg_data.columns
            if col not in ["SK_ID_CURR"] + count_cols
        ]
        for col in amount_cols:
            merged[col] = merged[col].fillna(merged[col].median())
        logger.info(f"Merged data shape: {merged.shape}")
        logger.info(f"New features added: {self.agg_data.shape[1] - 1}")

        return merged
