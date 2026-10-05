import os
import shutil
import sys
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame

from fraud_detection.cloud_storage.aws_storage import SimpleStorageService
from fraud_detection.configuration.aws_connection import aws_credentials_available
from fraud_detection.constants import (
    LOCAL_MODEL_REGISTRY_DIR,
    MODEL_STORAGE_ENV_KEY,
    MODEL_STORAGE_LOCAL,
    MODEL_STORAGE_S3,
)
from fraud_detection.entity.estimator import FraudDetectionModel
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging
from fraud_detection.utils.main_utils import load_object


class BaseModelRegistry(ABC):
    """
    Common contract for the production model store used by model evaluation,
    model pusher and the prediction pipeline.
    """

    storage: str

    @abstractmethod
    def is_model_present(self) -> bool:
        ...

    @abstractmethod
    def load_model(self) -> FraudDetectionModel:
        ...

    @abstractmethod
    def save_model(self, from_file: str, remove: bool = False) -> None:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        try:
            if getattr(self, "loaded_model", None) is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def predict_proba(self, dataframe: DataFrame) -> np.ndarray:
        try:
            if getattr(self, "loaded_model", None) is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict_proba(dataframe=dataframe)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e


class S3ModelRegistry(BaseModelRegistry):
    """
    Stores and retrieves the production model in an S3 bucket.
    """

    storage = MODEL_STORAGE_S3

    def __init__(self, bucket_name: str, model_path: str):
        """
        :param bucket_name: Name of the model bucket
        :param model_path: Key of the model inside the bucket
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: FraudDetectionModel = None

    def is_model_present(self) -> bool:
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=self.model_path)
        except Exception as e:
            logging.warning(f"Could not reach the S3 model registry: {e}")
            return False

    def load_model(self) -> FraudDetectionModel:
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        try:
            self.s3.upload_file(
                from_file, to_filename=self.model_path, bucket_name=self.bucket_name, remove=remove
            )
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def describe(self) -> str:
        return f"s3://{self.bucket_name}/{self.model_path}"


class LocalModelRegistry(BaseModelRegistry):
    """
    Stores and retrieves the production model on the local file system.

    Used as the default when no AWS credentials are configured, so that training,
    evaluation and prediction stay fully functional without a cloud account.
    """

    storage = MODEL_STORAGE_LOCAL

    def __init__(self, model_path: str, registry_dir: str = LOCAL_MODEL_REGISTRY_DIR):
        self.registry_dir = registry_dir
        self.model_path = os.path.join(registry_dir, model_path)
        self.loaded_model: FraudDetectionModel = None

    def is_model_present(self) -> bool:
        return os.path.exists(self.model_path)

    def load_model(self) -> FraudDetectionModel:
        try:
            if not self.is_model_present():
                raise FileNotFoundError(
                    f"No model found at {self.model_path}. Run the training pipeline first."
                )
            return load_object(file_path=self.model_path)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def save_model(self, from_file: str, remove: bool = False) -> None:
        try:
            os.makedirs(self.registry_dir, exist_ok=True)
            shutil.copyfile(from_file, self.model_path)
            logging.info(f"Saved the model to {self.model_path}")
            if remove:
                os.remove(from_file)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def describe(self) -> str:
        return os.path.abspath(self.model_path)


def get_model_registry(bucket_name: str, model_path: str) -> BaseModelRegistry:
    """
    Resolves the model registry backend.

    MODEL_STORAGE=s3 forces the S3 backend, MODEL_STORAGE=local forces the local one.
    When the variable is unset the S3 backend is selected only if AWS credentials are present.
    """
    requested = os.getenv(MODEL_STORAGE_ENV_KEY, "").strip().lower()

    if requested == MODEL_STORAGE_LOCAL:
        return LocalModelRegistry(model_path=model_path)

    if requested == MODEL_STORAGE_S3 or aws_credentials_available():
        return S3ModelRegistry(bucket_name=bucket_name, model_path=model_path)

    logging.info("AWS credentials not found, falling back to the local model registry")
    return LocalModelRegistry(model_path=model_path)
