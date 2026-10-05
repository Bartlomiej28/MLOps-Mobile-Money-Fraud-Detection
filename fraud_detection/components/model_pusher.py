import sys

from fraud_detection.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from fraud_detection.entity.config_entity import ModelPusherConfig
from fraud_detection.entity.model_registry import get_model_registry
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of model evaluation artifact stage
        :param model_pusher_config: Configuration for the model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.model_registry = get_model_registry(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   Publishes the accepted model to the configured model registry

        Output      :   Returns the model pusher artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info(f"Publishing the trained model to {self.model_registry.describe()}")

            self.model_registry.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                storage=self.model_registry.storage,
                model_path=self.model_registry.describe(),
            )

            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
