# download_tox21_deepchem.py

import deepchem as dc
import pandas as pd
import os

def main():
    # Use ECFP fingerprint featurizer (no deep learning needed)
    featurizer = dc.feat.CircularFingerprint(size=1024)

    # Load the dataset with the specified featurizer
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer, splitter='random')
    train, _, _ = datasets

    # Extract raw SMILES and labels for a single task (e.g., 'NR-AR')
    df = pd.DataFrame({
        'smiles': train.ids,
        'activity': train.y[:, 0]  # first target column
    })

    # Filter out rows with missing labels (-1 = unknown)
    df = df[df['activity'] != -1]

    # Save to CSV
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/drug_data.csv", index=False)
    print(f"Saved {len(df)} records to data/raw/drug_data.csv")

if __name__ == "__main__":
    main()
