import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from fraud_detection.constants import TARGET_COLUMN
from fraud_detection.data_access.fraud_data import FraudData
from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   Pulls the transaction data from the Hugging Face source and stores it
                        in the feature store of the current artifact run

        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Exporting data from the Hugging Face dataset")
            fraud_data = FraudData(
                n_parts=self.data_ingestion_config.n_parts,
                non_fraud_per_fraud=self.data_ingestion_config.non_fraud_per_fraud,
                random_state=self.data_ingestion_config.random_state,
                raw_data_file_path=self.data_ingestion_config.raw_data_file_path,
            )
            dataframe = fraud_data.export_data_as_dataframe()

            logging.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)

            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   Splits the dataframe into a stratified train set and test set

        Output      :   train and test csv files are written to the ingested directory
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of DataIngestion class")
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=self.data_ingestion_config.random_state,
                stratify=dataframe[TARGET_COLUMN],
            )
            logging.info("Performed stratified train test split on the dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            logging.info("Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file path.")
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion component of the training pipeline

        Output      :   train set and test set are returned as the artifacts of data ingestion
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from the Hugging Face dataset")

            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
