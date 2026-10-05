import os
import sys
from typing import List

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from fraud_detection.constants import (
    DATA_INGESTION_N_PARTS,
    DATA_INGESTION_NON_FRAUD_PER_FRAUD,
    DATA_INGESTION_RANDOM_STATE,
    HF_DATASET_FILE_TEMPLATE,
    HF_DATASET_REPO_ID,
    HF_DATASET_TOTAL_PARTS,
    HF_TOKEN_ENV_KEY,
    RAW_DATA_FILE_PATH,
    TARGET_COLUMN,
)
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging


class FraudData:
    """
    Builds the working dataset from the Hugging Face source dataset.

    The upstream dataset holds 21M transactions spread over 14 csv shards with a
    fraud rate of roughly 0.13 percent. A configurable number of shards is downloaded,
    every fraud row is kept and the legitimate rows are randomly down-sampled to the
    requested ratio, which keeps the pipeline runnable on a single machine.
    """

    def __init__(self,
                 repo_id: str = HF_DATASET_REPO_ID,
                 n_parts: int = DATA_INGESTION_N_PARTS,
                 non_fraud_per_fraud: int = DATA_INGESTION_NON_FRAUD_PER_FRAUD,
                 random_state: int = DATA_INGESTION_RANDOM_STATE,
                 raw_data_file_path: str = RAW_DATA_FILE_PATH):
        self.repo_id = repo_id
        self.n_parts = max(1, min(int(n_parts), HF_DATASET_TOTAL_PARTS))
        self.non_fraud_per_fraud = int(non_fraud_per_fraud)
        self.random_state = int(random_state)
        self.raw_data_file_path = raw_data_file_path

    def download_parts(self) -> List[str]:
        """
        Downloads the requested csv shards and returns their local paths.
        """
        try:
            token = os.getenv(HF_TOKEN_ENV_KEY) or None
            local_paths = []
            for part in range(1, self.n_parts + 1):
                filename = HF_DATASET_FILE_TEMPLATE.format(part=part)
                logging.info(f"Downloading {filename} from {self.repo_id}")
                local_paths.append(
                    hf_hub_download(
                        repo_id=self.repo_id,
                        filename=filename,
                        repo_type="dataset",
                        token=token,
                    )
                )
            return local_paths
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def build_dataset(self) -> pd.DataFrame:
        """
        Downloads the shards, applies the class-ratio sampling and caches the result.
        """
        try:
            frames = [pd.read_csv(path) for path in self.download_parts()]
            dataframe = pd.concat(frames, ignore_index=True)
            logging.info(f"Loaded {dataframe.shape[0]} raw transactions from {self.n_parts} shard(s)")

            dataframe = self.sample_dataset(dataframe)

            os.makedirs(os.path.dirname(self.raw_data_file_path), exist_ok=True)
            dataframe.to_csv(self.raw_data_file_path, index=False, header=True)
            logging.info(f"Cached the working dataset at {self.raw_data_file_path}")

            return dataframe
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def sample_dataset(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Keeps every fraud row and down-samples the legitimate rows to the configured ratio.
        """
        try:
            fraud_df = dataframe[dataframe[TARGET_COLUMN] == 1]
            legit_df = dataframe[dataframe[TARGET_COLUMN] == 0]

            n_legit = min(len(legit_df), len(fraud_df) * self.non_fraud_per_fraud)
            legit_df = legit_df.sample(n=n_legit, random_state=self.random_state)

            sampled = (
                pd.concat([fraud_df, legit_df], ignore_index=True)
                .sample(frac=1.0, random_state=self.random_state)
                .reset_index(drop=True)
            )

            logging.info(
                f"Sampled dataset: {sampled.shape[0]} rows, "
                f"{len(fraud_df)} fraud, {n_legit} legitimate"
            )
            return sampled
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def export_data_as_dataframe(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Returns the working dataset, reusing the local cache when it is available.
        """
        try:
            if not force_refresh and os.path.exists(self.raw_data_file_path):
                logging.info(f"Reading the cached dataset from {self.raw_data_file_path}")
                dataframe = pd.read_csv(self.raw_data_file_path)
            else:
                dataframe = self.build_dataset()

            dataframe = dataframe.replace({"na": np.nan})
            return dataframe
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
