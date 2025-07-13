import joblib
from sklearn.metrics import classification_report

def evaluate_model(X_test, y_test):
    model = joblib.load("models/random_forest_model.pkl")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
