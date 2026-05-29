# clasificar_mano.py
import numpy as np
from math import acos, degrees
from predict_gesture import predict_gesture


def _normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks = landmarks / max_dist
    return landmarks.flatten()


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return degrees(acos(np.clip(cos_angle, -1.0, 1.0)))


def _compute_angles(landmarks):
    l = np.array(landmarks)
    fingers = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    angles = []
    for f in fingers:
        angles.append(_angle(l[f[0]], l[f[1]], l[f[2]]))
        angles.append(_angle(l[f[1]], l[f[2]], l[f[3]]))
    return angles


def clasificar_mano(coordenadas):
    """
    Recibe las 21 coordenadas (x,y,z) de MediaPipe en orden
    y devuelve (gesto, confianza, probabilidades_por_clase).

    coordenadas: lista de 21 tuplas/listas (x, y, z)
                 o lista plana de 63 valores [x0,y0,z0, x1,y1,z1, ...]
    """
    coords = np.array(coordenadas, dtype=float)

    # Si viene como lista plana de 63 valores, reshape a (21, 3)
    if coords.ndim == 1:
        if len(coords) != 63:
            raise ValueError(f"Se esperan 63 valores (21x3), se recibieron {len(coords)}")
        coords = coords.reshape(21, 3)

    if coords.shape != (21, 3):
        raise ValueError(f"Se esperan 21 puntos con 3 coordenadas, se recibio shape {coords.shape}")

    landmarks = [tuple(row) for row in coords]

    norm = _normalize_landmarks(landmarks)
    angles = _compute_angles(landmarks)
    features = list(norm) + angles

    gesto, confianza, probs = predict_gesture(features)
    return gesto, confianza, probs


if __name__ == "__main__":
    # Ejemplo: coordenadas ficticias para probar
    coords_ejemplo = np.random.rand(21, 3).tolist()
    gesto, conf, probs = clasificar_mano(coords_ejemplo)
    print(f"Gesto: {gesto}")
    print(f"Confianza: {conf:.4f}")
    print(f"Probabilidades: {probs}")
