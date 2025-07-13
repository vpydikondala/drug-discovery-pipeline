import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return None

def load_and_featurize(path):
    df = pd.read_csv(path)
    features = []
    labels = []

    for i, row in df.iterrows():
        fp = featurize_smiles(row['smiles'])
        if fp:
            features.append(list(fp))
            labels.append(row['activity'])

    return pd.DataFrame(features), labels
