import os
import sys
from io import StringIO
from typing import List, Union

import dill
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket
from pandas import DataFrame, read_csv

from fraud_detection.configuration.aws_connection import S3Client
from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging


class SimpleStorageService:
    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def read_object(object_name, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str, bytes]:
        """
        Reads the body of the given S3 object.
        """
        logging.info("Entered the read_object method of SimpleStorageService class")
        try:
            body = object_name.get()["Body"].read()
            content = body.decode() if decode else body
            logging.info("Exited the read_object method of SimpleStorageService class")
            return StringIO(content) if make_readable else content
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Returns the bucket object for the given bucket name.
        """
        logging.info("Entered the get_bucket method of SimpleStorageService class")
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of SimpleStorageService class")
            return bucket
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Returns the object(s) stored under the given prefix in the given bucket.
        """
        logging.info("Entered the get_file_object method of SimpleStorageService class")
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
            logging.info("Exited the get_file_object method of SimpleStorageService class")
            return file_objects[0] if len(file_objects) == 1 else file_objects
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Loads and deserialises the model stored in the given bucket.
        """
        logging.info("Entered the load_model method of SimpleStorageService class")
        try:
            model_file = model_name if model_dir is None else f"{model_dir}/{model_name}"
            file_object = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(file_object, decode=False)
            model = dill.loads(model_obj)
            logging.info("Exited the load_model method of SimpleStorageService class")
            return model
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Creates the given folder in the given bucket when it does not exist yet.
        """
        logging.info("Entered the create_folder method of SimpleStorageService class")
        try:
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=f"{folder_name}/")
            logging.info("Exited the create_folder method of SimpleStorageService class")

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True) -> None:
        """
        Uploads a local file to the given bucket, optionally removing the local copy.
        """
        logging.info("Entered the upload_file method of SimpleStorageService class")
        try:
            logging.info(f"Uploading {from_filename} to {to_filename} in {bucket_name} bucket")
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(f"Uploaded {from_filename} to {to_filename} in {bucket_name} bucket")

            if remove:
                os.remove(from_filename)
                logging.info(f"Remove is set to {remove}, deleted the local file")

            logging.info("Exited the upload_file method of SimpleStorageService class")
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str,
                         bucket_name: str) -> None:
        """
        Writes the dataframe to a local csv file and uploads it to the given bucket.
        """
        logging.info("Entered the upload_df_as_csv method of SimpleStorageService class")
        try:
            data_frame.to_csv(local_filename, index=None, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
            logging.info("Exited the upload_df_as_csv method of SimpleStorageService class")
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:
        """
        Reads the given S3 object into a dataframe.
        """
        logging.info("Entered the get_df_from_object method of SimpleStorageService class")
        try:
            content = self.read_object(object_, make_readable=True)
            df = read_csv(content, na_values="na")
            logging.info("Exited the get_df_from_object method of SimpleStorageService class")
            return df
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """
        Reads a csv stored in the given bucket into a dataframe.
        """
        logging.info("Entered the read_csv method of SimpleStorageService class")
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of SimpleStorageService class")
            return df
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
