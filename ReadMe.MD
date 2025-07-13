# Drug Discovery Data Pipeline – DevOps for ML

## Overview
This project builds an end-to-end pipeline for drug-likeness prediction using cheminformatics features and machine learning. It includes data processing, model training, and prediction components, with CI/CD pipeline support for deployment using Azure DevOps and Docker.

## Setup

1. Clone repo
2. Create and activate Python environment
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Download and place raw data in `data/raw/drug_data.csv` (sample dataset link below)

## Usage

- Run data processing and feature extraction
python src/data_processing.py

- Train model
python src/model.py

- Predict drug-likeness from SMILES
python src/predict.py "CC(=O)OC1=CC=CC=C1C(=O)O"

## Docker

Build and run container:

docker build -t drug-predictor .
docker run drug-predictor "CC(=O)OC1=CC=CC=C1C(=O)O"

## CI/CD Pipeline

The `pipelines/azure-pipelines.yml` file defines an Azure DevOps pipeline for automated testing and model training.

## Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn, rdkit-pypi, joblib, mlflow

## Data Source

Sample drug-likeness dataset can be found here:

- [ChEMBL or curated dataset with SMILES and activity labels]

## Tools & Technologies:

- Python, Pandas, Numpy, scikit-learn  
- RDKit for cheminformatics  
- MLflow for model tracking  
- Docker for containerization  
- Azure Data Factory, Databricks, Azure DevOps (simulated locally here)  