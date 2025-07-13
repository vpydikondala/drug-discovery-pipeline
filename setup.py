from setuptools import setup, find_packages

setup(
    name='drug_discovery_pipeline',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'rdkit-pypi',
        'joblib',
        'mlflow'
    ],
)
