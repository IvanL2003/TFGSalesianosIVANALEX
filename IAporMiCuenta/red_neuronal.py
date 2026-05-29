import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

data = pd.read_csv("hand_dataset.csv")

X = data.drop("label", axis=1)
y = data["label"]

# GUARDAR ORDEN DE FEATURES
joblib.dump(X.columns.tolist(), "feature_names.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "model.pkl")
