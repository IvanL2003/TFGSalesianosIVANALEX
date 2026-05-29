# predict_gesture.py
import numpy as np
import joblib
import pandas as pd

# =========================
# Cargar modelo entrenado
# =========================
model = joblib.load("model.pkl")
FEATURE_NAMES = joblib.load("feature_names.pkl")


# =========================
# Función pública de predicción
# =========================
def predict_gesture(features):
    df = pd.DataFrame([features], columns=FEATURE_NAMES)

    prediction = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    confidence = float(np.max(probs))
    return prediction, confidence, dict(zip(model.classes_, probs))




# =========================
# TEST MANUAL
# =========================
if __name__ == "__main__":
    fake_input = np.random.rand(73)  # ejemplo fake
    pred, conf, all_probs = predict_gesture(fake_input)

    print("Gesto:", pred)
    print("Confianza:", conf)
    print("Todas:", all_probs)
