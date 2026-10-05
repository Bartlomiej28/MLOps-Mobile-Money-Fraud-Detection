import os
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pandas import DataFrame

from fraud_detection.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from fraud_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from fraud_detection.entity.config_entity import DataValidationConfig
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging
from fraud_detection.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column count present: [{status}]")
            return status
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of the numerical and categorical columns

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns

            missing_numerical_columns = [
                column for column in self._schema_config["numerical_columns"]
                if column not in dataframe_columns
            ]
            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            missing_categorical_columns = [
                column for column in self._schema_config["categorical_columns"]
                if column not in dataframe_columns
            ]
            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   Builds an Evidently data drift report, persists it as yaml and html
                        and decides whether the dataset as a whole has drifted

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            report = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(
                current_data=current_df.drop(columns=[TARGET_COLUMN], errors="ignore"),
                reference_data=reference_df.drop(columns=[TARGET_COLUMN], errors="ignore"),
            )

            report_dict = snapshot.dict()

            n_features = 0
            n_drifted_features = 0
            drift_share = 0.0
            column_drift = {}

            for metric in report_dict.get("metrics", []):
                config = metric.get("config", {})
                metric_type = config.get("type", "")
                value = metric.get("value")

                if metric_type.endswith("DriftedColumnsCount") and isinstance(value, dict):
                    n_drifted_features = int(value.get("count", 0))
                    drift_share = float(value.get("share", 0.0))
                elif metric_type.endswith("ValueDrift"):
                    column = config.get("column")
                    threshold = float(config.get("threshold", 0.05))
                    method = str(config.get("method", ""))
                    drifted = value < threshold if "p_value" in method else value > threshold
                    column_drift[column] = {
                        "method": method,
                        "score": float(value),
                        "threshold": threshold,
                        "drift_detected": bool(drifted),
                    }
                    n_features += 1

            drift_status = drift_share >= self.data_validation_config.drift_share_threshold

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content={
                    "n_features": n_features,
                    "n_drifted_features": n_drifted_features,
                    "drift_share": drift_share,
                    "drift_share_threshold": self.data_validation_config.drift_share_threshold,
                    "dataset_drift": bool(drift_status),
                    "columns": column_drift,
                },
                replace=True,
            )

            html_report_path = self.data_validation_config.drift_report_file_path.replace(".yaml", ".html")
            os.makedirs(os.path.dirname(html_report_path), exist_ok=True)
            snapshot.save_html(html_report_path)

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            return drift_status
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline

        Output      :   Returns the data validation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            train_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            if not self.validate_number_of_columns(dataframe=train_df):
                validation_error_msg += "Columns are missing in training dataframe. "
            if not self.validate_number_of_columns(dataframe=test_df):
                validation_error_msg += "Columns are missing in test dataframe. "
            if not self.is_column_exist(df=train_df):
                validation_error_msg += "Required columns are missing in training dataframe. "
            if not self.is_column_exist(df=test_df):
                validation_error_msg += "Required columns are missing in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                validation_error_msg = "Drift detected" if drift_status else "Drift not detected"
            else:
                logging.info(f"Validation error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
