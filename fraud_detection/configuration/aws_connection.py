import os

import boto3

from fraud_detection.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    REGION_NAME,
)


def aws_credentials_available() -> bool:
    """
    Returns True when both AWS credential environment variables are populated.
    """
    return bool(os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)) and bool(os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY))


class S3Client:
    s3_client = None
    s3_resource = None

    def __init__(self, region_name: str = None):
        """
        Reads the AWS credentials from the environment and opens a connection to S3.
        Raises when the environment variables are not set.
        """
        region_name = region_name or os.getenv("AWS_DEFAULT_REGION", REGION_NAME)

        if S3Client.s3_resource is None or S3Client.s3_client is None:
            access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            if access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set.")
            if secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")

            S3Client.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name,
            )
            S3Client.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name,
            )

        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
