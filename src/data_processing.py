import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.dropna(subset=['smiles', 'activity'])
    df = df[df['activity'].isin([0,1])]  # Binary classification (drug-like or not)
    return df

def featurize(df):
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['MolWt'] = df['mol'].apply(Descriptors.MolWt)
    df['TPSA'] = df['mol'].apply(Descriptors.TPSA)
    df['NumRotatableBonds'] = df['mol'].apply(Descriptors.NumRotatableBonds)

    df['fingerprints'] = df['mol'].apply(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
    
    fps = list(df['fingerprints'])
    X_fp = np.array([list(fp) for fp in fps])
    
    descriptors = df[['MolWt', 'TPSA', 'NumRotatableBonds']].values
    X = np.hstack([descriptors, X_fp])
    y = df['activity'].values
    return X, y

def main():
    raw_path = os.path.join('data', 'raw', 'drug_data.csv')
    df = load_data(raw_path)
    df_clean = clean_data(df)
    X, y = featurize(df_clean)
    np.save(os.path.join('data', 'processed', 'X.npy'), X)
    np.save(os.path.join('data', 'processed', 'y.npy'), y)
    print("Data processing completed. Features and labels saved.")

if __name__ == "__main__":
    main()
