"""Setup configuration for DeepCTR project."""

from setuptools import find_packages, setup

setup(
    name="deepctr",
    version="0.1.0",
    description="DeepCTR - Click-Through Rate Prediction with DeepFM",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.1",
        "pyarrow>=13.0",
        "scikit-learn>=1.3",
        "torch>=2.1",
        "joblib>=1.3",
    ],
)





