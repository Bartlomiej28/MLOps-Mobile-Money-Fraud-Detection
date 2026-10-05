import argparse
import os
import sys

from fraud_detection.entity.config_entity import ModelPusherConfig
from fraud_detection.entity.model_registry import get_model_registry
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging
from fraud_detection.pipeline.training_pipeline import TrainPipeline


def seed_model_registry(local_model_path: str = None) -> str:
    """
    Makes sure the resolved model registry holds a production model.

    An existing model is left untouched. When a local model file is given it is
    published as is, otherwise the whole training pipeline runs and publishes its result.
    """
    try:
        config = ModelPusherConfig()
        registry = get_model_registry(bucket_name=config.bucket_name, model_path=config.s3_model_key_path)

        if registry.is_model_present():
            logging.info(f"Model registry already holds a model at {registry.describe()}")
            return "present"

        if local_model_path and os.path.exists(local_model_path):
            logging.info(f"Seeding {registry.describe()} with {local_model_path}")
            registry.save_model(from_file=local_model_path)
            return "uploaded"

        logging.info(f"Model registry {registry.describe()} is empty, running the training pipeline")
        TrainPipeline().run_pipeline()
        return "trained"
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the model registry when it is empty")
    parser.add_argument("--model", default=None, help="local model.pkl to publish instead of training")
    args = parser.parse_args()
    outcome = seed_model_registry(local_model_path=args.model)
    print(f"model registry: {outcome}")
