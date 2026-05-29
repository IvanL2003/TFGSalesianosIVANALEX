# grafico_mano.py
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from clasificar_mano import clasificar_mano

mp_hands = mp.solutions.hands.Hands()

# Conexiones de MediaPipe para dibujar los "huesos"
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),       # indice
    (0, 9), (9, 10), (10, 11), (11, 12),   # medio
    (0, 13), (13, 14), (14, 15), (15, 16), # anular
    (0, 17), (17, 18), (18, 19), (19, 20), # menique
    (5, 9), (9, 13), (13, 17),             # palma
]


def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks = landmarks / max_dist
    return landmarks


def main():
    cap = cv2.VideoCapture(0)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 7))

    # Crear los objetos graficos una sola vez
    lines = []
    for _ in CONNECTIONS:
        line, = ax.plot([], [], "deepskyblue", linewidth=2)
        lines.append(line)

    points, = ax.plot([], [], "o", color="white", markersize=6, markeredgecolor="deepskyblue")
    wrist_point, = ax.plot([], [], "o", color="red", markersize=8)
    title_text = ax.set_title("", fontsize=14, color="white", fontweight="bold")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.grid(True, alpha=0.15, color="white")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_color("gray")

    print("Pulsa 'q' en la ventana de matplotlib para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks_raw = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            norm = normalize_landmarks(landmarks_raw)

            # x e y normalizados (invertir y para que los dedos apunten arriba)
            xs = norm[:, 0]
            ys = -norm[:, 1]

            # Actualizar lineas
            for i, (a, b) in enumerate(CONNECTIONS):
                lines[i].set_data([xs[a], xs[b]], [ys[a], ys[b]])

            # Actualizar puntos
            points.set_data(xs, ys)
            wrist_point.set_data([xs[0]], [ys[0]])

            # Prediccion
            gesto, conf, _ = clasificar_mano(landmarks_raw)
            title_text.set_text(f"{gesto}  ({conf:.0%})")
        else:
            # Sin mano detectada: limpiar
            for line in lines:
                line.set_data([], [])
            points.set_data([], [])
            wrist_point.set_data([], [])
            title_text.set_text("No se detecta mano")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Detectar cierre de ventana o tecla 'q'
        if not plt.fignum_exists(fig.number):
            break
        if plt.waitforbuttonpress(0.001):
            break

    cap.release()
    plt.close(fig)


if __name__ == "__main__":
    main()
