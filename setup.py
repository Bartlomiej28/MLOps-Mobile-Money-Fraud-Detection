from setuptools import find_packages, setup

setup(
    name="fraud_detection",
    version="1.0.0",
    author="Bartlomiej28",
    author_email="bartlomiej.boczylo@gmail.com",
    description="Production ready MLOps pipeline for mobile money fraud detection",
    packages=find_packages(),
    python_requires=">=3.10",
)
