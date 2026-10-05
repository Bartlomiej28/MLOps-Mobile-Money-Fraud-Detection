import argparse

from fraud_detection.constants import (
    DATA_INGESTION_N_PARTS,
    DATA_INGESTION_NON_FRAUD_PER_FRAUD,
    DATA_INGESTION_RANDOM_STATE,
    HF_DATASET_REPO_ID,
    RAW_DATA_FILE_PATH,
    TARGET_COLUMN,
)
from fraud_detection.data_access.fraud_data import FraudData


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Build the working fraud detection dataset from {HF_DATASET_REPO_ID}"
    )
    parser.add_argument("--parts", type=int, default=DATA_INGESTION_N_PARTS,
                        help="number of csv shards to download (1-14)")
    parser.add_argument("--non-fraud-per-fraud", type=int, default=DATA_INGESTION_NON_FRAUD_PER_FRAUD,
                        help="how many legitimate rows to keep per fraud row")
    parser.add_argument("--random-state", type=int, default=DATA_INGESTION_RANDOM_STATE)
    parser.add_argument("--output", type=str, default=RAW_DATA_FILE_PATH)
    args = parser.parse_args()

    fraud_data = FraudData(
        n_parts=args.parts,
        non_fraud_per_fraud=args.non_fraud_per_fraud,
        random_state=args.random_state,
        raw_data_file_path=args.output,
    )
    dataframe = fraud_data.build_dataset()

    print(f"Saved {dataframe.shape[0]} rows and {dataframe.shape[1]} columns to {args.output}")
    print(dataframe[TARGET_COLUMN].value_counts().to_string())


if __name__ == "__main__":
    main()
