import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from fraud_detection.constants import TARGET_COLUMN
from fraud_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from fraud_detection.entity.config_entity import ModelEvaluationConfig
from fraud_detection.entity.model_registry import BaseModelRegistry, get_model_registry
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging


@dataclass
class EvaluateModelResponse:
    trained_model_score: float
    best_model_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_registry = get_model_registry(
                bucket_name=model_eval_config.bucket_name,
                model_path=model_eval_config.s3_model_key_path,
            )
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_best_model(self) -> Optional[BaseModelRegistry]:
        """
        Method Name :   get_best_model
        Description :   Returns the registry handle when a production model is available

        Output      :   Returns the model registry object or None
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.model_registry.is_model_present():
                return self.model_registry
            return None
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def score_production_model(best_model: BaseModelRegistry, x: pd.DataFrame, y: pd.Series) -> float:
        """
        Scores the production model with ROC AUC on probabilities and falls back to F1
        on the hard predictions when the model does not expose probabilities.
        """
        try:
            probabilities = best_model.predict_proba(x)[:, 1]
            return float(roc_auc_score(y, probabilities))
        except Exception as e:
            logging.warning(f"Production model has no usable probabilities, scoring with f1: {e}")
            return float(f1_score(y, best_model.predict(x), zero_division=0))

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   Compares the freshly trained model with the production model
                        on the untouched test set using ROC AUC

        Output      :   Returns the evaluation response
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN].astype(int)

            trained_model_score = self.model_trainer_artifact.metric_artifact.roc_auc_score

            best_model_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Found a production model at {best_model.describe()}")
                best_model_score = self.score_production_model(best_model, x, y)

            tmp_best_model_score = 0.0 if best_model_score is None else best_model_score
            difference = trained_model_score - tmp_best_model_score

            result = EvaluateModelResponse(
                trained_model_score=trained_model_score,
                best_model_score=best_model_score,
                is_model_accepted=difference > self.model_eval_config.changed_threshold_score,
                difference=difference,
            )
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function initiates all the steps of the model evaluation

        Output      :   Returns the model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                model_path=self.model_registry.describe(),
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_score=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
