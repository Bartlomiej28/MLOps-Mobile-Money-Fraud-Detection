import sys
from typing import Tuple

import numpy as np
from neuro_mf import ModelFactory
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fraud_detection.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from fraud_detection.entity.config_entity import ModelTrainerConfig
from fraud_detection.entity.estimator import FraudDetectionModel
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging
from fraud_detection.utils.main_utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
        """
        Returns the probability threshold that maximises F1 together with that F1 value.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        best = int(np.nanargmax(f1))
        return float(thresholds[best]), float(f1[best])

    def select_threshold(self, model_obj: object, x_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Method Name :   select_threshold
        Description :   Picks the decision threshold on out-of-fold probabilities of the
                        training set so the test set stays untouched

        Output      :   Returns the selected threshold
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if not hasattr(model_obj, "predict_proba"):
                logging.info("Model has no predict_proba, keeping the default threshold")
                return self.model_trainer_config.default_threshold

            cv = StratifiedKFold(
                n_splits=self.model_trainer_config.threshold_cv_folds,
                shuffle=True,
                random_state=self.model_trainer_config.random_state,
            )
            oof_probabilities = cross_val_predict(
                clone(model_obj), x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]

            threshold, oof_f1 = self.best_f1_threshold(y_train, oof_probabilities)
            logging.info(
                f"Out-of-fold roc_auc: {roc_auc_score(y_train, oof_probabilities):.4f}, "
                f"selected threshold {threshold:.4f} with out-of-fold f1 {oof_f1:.4f}"
            )
            return threshold
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_model_object_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[object, float, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   Uses neuro_mf to run the configured grid search, selects the decision
                        threshold on the training set and measures the metrics on the
                        untouched test set

        Output      :   Returns the best model detail, the threshold and the metric artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get the best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)

            x_train, y_train = train[:, :-1], train[:, -1].astype(int)
            x_test, y_test = test[:, :-1], test[:, -1].astype(int)

            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_score
            )
            model_obj = best_model_detail.best_model
            logging.info(f"Best model: {type(model_obj).__name__}, cross validated score {best_model_detail.best_score:.4f}")

            threshold = self.select_threshold(model_obj, x_train, y_train)

            if hasattr(model_obj, "predict_proba"):
                probabilities = model_obj.predict_proba(x_test)[:, 1]
                y_pred = (probabilities >= threshold).astype(int)
                roc_auc = roc_auc_score(y_test, probabilities)
                average_precision = average_precision_score(y_test, probabilities)
            else:
                y_pred = model_obj.predict(x_test)
                roc_auc = roc_auc_score(y_test, y_pred)
                average_precision = average_precision_score(y_test, y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            logging.info(
                f"Test metrics -> accuracy: {accuracy:.4f}, f1: {f1:.4f}, "
                f"precision: {precision:.4f}, recall: {recall:.4f}, roc_auc: {roc_auc:.4f}, "
                f"average_precision: {average_precision:.4f}, threshold: {threshold:.4f}"
            )

            metric_artifact = ClassificationMetricArtifact(
                f1_score=float(f1),
                precision_score=float(precision),
                recall_score=float(recall),
                roc_auc_score=float(roc_auc),
                average_precision_score=float(average_precision),
                threshold=float(threshold),
            )

            return best_model_detail, threshold, metric_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model trainer steps

        Output      :   Returns the model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, threshold, metric_artifact = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_score:
                logging.info("No best model found with score more than the base score")
                raise Exception("No best model found with score more than the base score")

            fraud_detection_model = FraudDetectionModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model,
                threshold=threshold,
            )
            logging.info(f"Created the fraud detection model object {fraud_detection_model} with preprocessor and model")

            save_object(self.model_trainer_config.trained_model_file_path, fraud_detection_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
