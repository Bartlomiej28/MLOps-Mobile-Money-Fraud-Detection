import sys

import numpy as np
from pandas import DataFrame

from fraud_detection.entity.config_entity import FraudDetectionPredictorConfig
from fraud_detection.entity.estimator import TargetValueMapping
from fraud_detection.entity.model_registry import get_model_registry
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]


class FraudTransactionData:
    def __init__(self,
                 step,
                 type,
                 amount,
                 oldbalanceOrg,
                 newbalanceOrig,
                 oldbalanceDest,
                 newbalanceDest):
        """
        Holds a single raw transaction coming from the web form or the json api.
        """
        try:
            self.step = step
            self.type = type
            self.amount = amount
            self.oldbalanceOrg = oldbalanceOrg
            self.newbalanceOrig = newbalanceOrig
            self.oldbalanceDest = oldbalanceDest
            self.newbalanceDest = newbalanceDest
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_input_data_frame(self) -> DataFrame:
        """
        Returns a single row DataFrame matching the raw schema expected by the model.
        """
        try:
            return DataFrame(self.get_data_as_dict())
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_data_as_dict(self) -> dict:
        """
        Returns the transaction as a dictionary with the model input dtypes applied.
        """
        logging.info("Entered get_data_as_dict method of FraudTransactionData class")
        try:
            input_data = {
                "step": [int(self.step)],
                "type": [str(self.type)],
                "amount": [float(self.amount)],
                "oldbalanceOrg": [float(self.oldbalanceOrg)],
                "newbalanceOrig": [float(self.newbalanceOrig)],
                "oldbalanceDest": [float(self.oldbalanceDest)],
                "newbalanceDest": [float(self.newbalanceDest)],
            }
            logging.info("Exited get_data_as_dict method of FraudTransactionData class")
            return input_data
        except Exception as e:
            raise FraudDetectionException(e, sys) from e


class FraudDetectionClassifier:
    def __init__(self, prediction_pipeline_config: FraudDetectionPredictorConfig = FraudDetectionPredictorConfig()):
        """
        :param prediction_pipeline_config: Configuration used to locate the production model
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.model_registry = get_model_registry(
                bucket_name=prediction_pipeline_config.model_bucket_name,
                model_path=prediction_pipeline_config.model_file_path,
            )
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Returns the predicted class for every row of the given dataframe.
        """
        try:
            logging.info("Entered predict method of FraudDetectionClassifier class")
            return self.model_registry.predict(dataframe)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def predict_with_probability(self, dataframe: DataFrame) -> tuple:
        """
        Returns the predicted label and the fraud probability of the first row.
        """
        try:
            prediction = int(self.model_registry.predict(dataframe)[0])
            label = TargetValueMapping().reverse_mapping()[prediction]

            probability = None
            try:
                probability = float(self.model_registry.predict_proba(dataframe)[0][1])
            except Exception as e:
                logging.warning(f"Probability is not available for the loaded model: {e}")

            return prediction, label, probability
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
