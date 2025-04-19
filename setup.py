from setuptools import setup, find_packages

setup(
    name="learning_center_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.2",
        "jupyter>=1.0.0"
    ],
) 