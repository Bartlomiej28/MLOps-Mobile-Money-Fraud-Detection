import os

PIPELINE_NAME: str = "fraud_detection"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME: str = "fraud_transactions.csv"
MODEL_FILE_NAME: str = "model.pkl"

TARGET_COLUMN: str = "isFraud"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")

HF_DATASET_REPO_ID: str = "CiferAI/Cifer-Fraud-Detection-Dataset-AF"
HF_DATASET_FILE_TEMPLATE: str = "Cifer-Fraud-Detection-Dataset-AF-part-{part}-14.csv"
HF_DATASET_TOTAL_PARTS: int = 14
HF_TOKEN_ENV_KEY: str = "HUGGINGFACE_TOKEN"

RAW_DATA_DIR: str = "data"
RAW_DATA_FILE_PATH: str = os.path.join(RAW_DATA_DIR, FILE_NAME)

AWS_ACCESS_KEY_ID_ENV_KEY: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY: str = "AWS_SECRET_ACCESS_KEY"
REGION_NAME: str = "us-east-1"

MODEL_STORAGE_ENV_KEY: str = "MODEL_STORAGE"
MODEL_STORAGE_S3: str = "s3"
MODEL_STORAGE_LOCAL: str = "local"
LOCAL_MODEL_REGISTRY_DIR: str = "saved_models"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_N_PARTS: int = 2
DATA_INGESTION_NON_FRAUD_PER_FRAUD: int = 20
DATA_INGESTION_RANDOM_STATE: int = 42


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DRIFT_SHARE_THRESHOLD: float = 0.5


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.5
MODEL_TRAINER_THRESHOLD_CV_FOLDS: int = 5
MODEL_TRAINER_DEFAULT_THRESHOLD: float = 0.5
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


"""
MODEL EVALUATION related constant
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.0
MODEL_BUCKET_NAME: str = os.getenv("MODEL_BUCKET_NAME", "fraud-detection-model-registry")


APP_HOST: str = "0.0.0.0"
APP_PORT: int = 8080
