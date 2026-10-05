import os
import sys

import dill
import numpy as np
import yaml
from pandas import DataFrame

from fraud_detection.exception import FraudDetectionException
from fraud_detection.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info("Exited the load_object method of utils")
        return obj
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.ndarray data to save
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return: np.ndarray data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop the columns from a pandas DataFrame, ignoring the ones that are absent
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns method of utils")
    try:
        existing = [col for col in cols if col in df.columns]
        df = df.drop(columns=existing, axis=1)
        logging.info("Exited the drop_columns method of utils")
        return df
    except Exception as e:
        raise FraudDetectionException(e, sys) from e
