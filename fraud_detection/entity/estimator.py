import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.Legitimate: int = 0
        self.Fraud: int = 1

    def _asdict(self) -> dict:
        return self.__dict__

    def reverse_mapping(self) -> dict:
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives the modelling features from the raw transaction columns.

    It is the first step of the persisted preprocessing pipeline, so exactly the same
    feature derivation is applied during training, evaluation and online prediction.
    """

    def __init__(self, drop_columns: list = None, amount_band_edges: list = None,
                 amount_band_labels: list = None):
        self.drop_columns = drop_columns
        self.amount_band_edges = amount_band_edges
        self.amount_band_labels = amount_band_labels

    def fit(self, X: DataFrame, y=None):
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        try:
            df = X.copy()

            df["hour_of_day"] = df["step"] % 24
            df["day_of_month"] = df["step"] // 24

            df["balance_change_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
            df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

            df["error_balance_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
            df["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

            df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)

            df["orig_balance_emptied"] = np.where(
                (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0), "Y", "N"
            )
            df["dest_previously_empty"] = np.where(df["oldbalanceDest"] == 0, "Y", "N")

            df["amount_band"] = self._build_amount_band(df["amount"])

            return df.drop(columns=[c for c in (self.drop_columns or []) if c in df.columns])

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def _build_amount_band(self, amount: pd.Series) -> np.ndarray:
        edges = list(self.amount_band_edges or [10000, 100000, 1000000])
        labels = list(self.amount_band_labels or ["low", "medium", "high", "very_high"])
        bins = [-np.inf] + edges + [np.inf]
        return pd.cut(amount, bins=bins, labels=labels, right=True).astype(str).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return None


class FraudDetectionModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object,
                 threshold: float = 0.5):
        """
        :param preprocessing_object: Input Object of preprocessor
        :param trained_model_object: Input Object of trained model
        :param threshold: Fraud probability above which a transaction is labelled as fraud
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.threshold = threshold

    @property
    def decision_threshold(self) -> float:
        return float(getattr(self, "threshold", 0.5))

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Accepts raw transaction rows, transforms them with the persisted preprocessing
        pipeline and returns the class predictions of the trained model. When the model
        exposes probabilities the persisted decision threshold is applied to them.
        """
        logging.info("Entered predict method of FraudDetectionModel class")
        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)
            logging.info("Used the trained model to get predictions")
            if hasattr(self.trained_model_object, "predict_proba"):
                probabilities = self.trained_model_object.predict_proba(transformed_feature)[:, 1]
                return (probabilities >= self.decision_threshold).astype(int)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def predict_proba(self, dataframe: DataFrame) -> np.ndarray:
        """
        Returns the fraud probability of every row when the trained model exposes it.
        """
        try:
            if not hasattr(self.trained_model_object, "predict_proba"):
                raise AttributeError(
                    f"{type(self.trained_model_object).__name__} does not expose predict_proba"
                )
            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict_proba(transformed_feature)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}(threshold={self.decision_threshold:.3f})"

    def __str__(self):
        return self.__repr__()
