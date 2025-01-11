
from setuptools import setup, find_packages

setup(
    name="TR_DS_ChoudharyOm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "transformers",
        "scikit-learn",
        "matplotlib",
        "torch"
    ],
    description="Fine-tuning BERT for multi-label classification.",
    author="Choudhary Om",
    license="MIT",
)
