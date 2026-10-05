import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from fraud_detection.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from fraud_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from fraud_detection.entity.config_entity import DataTransformationConfig
from fraud_detection.entity.estimator import FraudFeatureEngineer
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging
from fraud_detection.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        :param data_validation_artifact: Output reference of data validation artifact stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   Builds the preprocessing pipeline: feature engineering followed by
                        encoding, power transformation and scaling of the derived columns

        Output      :   data transformer object is created and returned
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            oh_columns = self._schema_config["oh_columns"]
            or_columns = self._schema_config["or_columns"]
            transform_columns = self._schema_config["transform_columns"]
            num_features = self._schema_config["num_features"]
            ordinal_categories = self._schema_config["ordinal_categories"]

            feature_engineer = FraudFeatureEngineer(
                drop_columns=self._schema_config["drop_columns"],
                amount_band_edges=self._schema_config["amount_band_edges"],
                amount_band_labels=ordinal_categories["amount_band"],
            )

            oh_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ordinal_encoder = OrdinalEncoder(
                categories=[list(ordinal_categories[column]) for column in or_columns],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            numeric_transformer = StandardScaler()

            logging.info("Initialized OneHotEncoder, OrdinalEncoder, StandardScaler and PowerTransformer")

            transform_pipe = Pipeline(steps=[("transformer", PowerTransformer(method="yeo-johnson"))])

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features),
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")

            return Pipeline(steps=[("feature_engineer", feature_engineer), ("preprocessor", preprocessor)])
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   Fits the preprocessing pipeline on the training data and persists the
                        transformed arrays. The class imbalance is left untouched and handled
                        by the class weights of the estimators declared in config/model.yaml

        Output      :   data transformer steps are performed and preprocessor object is created
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].astype(int)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].astype(int)

            logging.info("Got input features and target feature of the training and testing dataset")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Used the preprocessor object to fit transform the train features")

            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Used the preprocessor object to transform the test features")

            fraud_share = float(target_feature_train_df.mean())
            logging.info(
                f"Training set kept imbalanced: {len(target_feature_train_df)} rows, "
                f"fraud share {fraud_share:.4f}, class weights are applied by the estimators"
            )

            logging.info("Test dataset is left untouched to keep the evaluation metrics unbiased")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved the preprocessor object and the transformed arrays")
            logging.info("Exited initiate_data_transformation method of DataTransformation class")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
