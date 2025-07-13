import sys
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import os

def featurize_single(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = [Descriptors.MolWt(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol)]
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_array = np.array(list(fp))
    features = np.hstack([descriptors, fp_array])
    return features.reshape(1, -1)

def predict(smiles):
    model_path = os.path.join('model', 'drug_likeness_rf.pkl')
    model = joblib.load(model_path)
    features = featurize_single(smiles)
    pred = model.predict(features)
    print(f"Prediction (drug-likeness) for {smiles}: {pred[0]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <SMILES_STRING>")
        sys.exit(1)
    smiles = sys.argv[1]
    predict(smiles)
