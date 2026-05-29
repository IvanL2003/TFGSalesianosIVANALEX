# camera_landmarks.py
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from predict_gesture import predict_gesture

mp_hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils


# =========================
# Normalizar landmarks
# =========================
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks = landmarks / max_dist

    return landmarks.flatten()


# =========================
# Ángulo entre 3 puntos
# =========================
def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return degrees(acos(np.clip(cos_angle, -1.0, 1.0)))


# =========================
# Calcular ángulos dedos
# =========================
def compute_angles(landmarks):
    l = np.array(landmarks)

    fingers = [
        [1,2,3,4],   # thumb
        [5,6,7,8],   # index
        [9,10,11,12],# middle
        [13,14,15,16],# ring
        [17,18,19,20] # pinky
    ]

    angles = []
    for f in fingers:
        angles.append(angle(l[f[0]], l[f[1]], l[f[2]]))
        angles.append(angle(l[f[1]], l[f[2]], l[f[3]]))

    return angles


# =========================
# Webcam loop
# =========================
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]

            norm = normalize_landmarks(landmarks)
            angles = compute_angles(landmarks)

            features = list(norm) + angles

            # IA prediction
            gesture, conf, probs = predict_gesture(features)

            # Mostrar en pantalla
            text = f"{gesture} ({conf:.2f})"
            cv2.putText(frame, text, (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture AI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
