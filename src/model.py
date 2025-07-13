import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

def train_model():
    X = np.load(os.path.join('data', 'processed', 'X.npy'))
    y = np.load(os.path.join('data', 'processed', 'y.npy'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Test F1 Score: {f1_score(y_test, y_pred):.3f}")

    os.makedirs('model', exist_ok=True)
    joblib.dump(best_model, os.path.join('model', 'drug_likeness_rf.pkl'))
    print("Model saved.")
    return best_model

if __name__ == "__main__":
    train_model()
