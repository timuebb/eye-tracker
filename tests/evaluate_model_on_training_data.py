import os
import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import time
from utils.eye_utils import extract_eye_from_landmarks, estimate_head_pose
import mediapipe as mp
from datetime import datetime

# Konfiguration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fold1_dual_model.keras")
IMG_SIZE = (64, 64)
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]
DISPLAY_W, DISPLAY_H = pyautogui.size()
DPI = 110  # Bildschirm-DPI, ggf. anpassen für echte cm-Angabe

# Modell laden
model = tf.keras.models.load_model(MODEL_PATH)
expects_pose = "head_pose" in [inp.name.split(":")[0] for inp in model.inputs]

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

cap = cv2.VideoCapture(1)

# Testpunkte im Raster
GRID_SIZE = 5
WAIT_SEC = 2.0  # Wartezeit je Punkt
points = [
    (int(x * DISPLAY_W / (GRID_SIZE - 1)), int(y * DISPLAY_H / (GRID_SIZE - 1)))
    for y in range(GRID_SIZE)
    for x in range(GRID_SIZE)
]

errors = []


def get_landmarks_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None


def show_target_and_wait(x, y, seconds):
    screen = np.ones((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8) * 255
    cv2.circle(screen, (x, y), 20, (0, 0, 255), -1)
    cv2.putText(
        screen,
        "Schaue auf den Punkt...",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
    )

    cv2.namedWindow("Kalibrierung", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Kalibrierung", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    start = time.time()
    while time.time() - start < seconds:
        cv2.imshow("Kalibrierung", screen)
        if cv2.waitKey(1) & 0xFF == 27:
            return False
    return True


# Ablauf
for i, (x, y) in enumerate(points):
    print(f"Punkt {i+1}/25 bei ({x},{y})")
    if not show_target_and_wait(x, y, WAIT_SEC):
        break

    # Bild erfassen
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Kein Kamerabild")
        continue

    landmarks = get_landmarks_from_frame(frame)
    if landmarks is None:
        print("⚠️ Keine Landmarks erkannt")
        continue

    face_box = None
    left_eye = extract_eye_from_landmarks(
        frame, landmarks, LEFT_EYE, IMG_SIZE, face_box
    )
    right_eye = extract_eye_from_landmarks(
        frame, landmarks, RIGHT_EYE, IMG_SIZE, face_box
    )

    if left_eye is None or right_eye is None:
        print("⚠️ Augen konnten nicht extrahiert werden")
        continue

    inputs = {
        "left_eye": np.expand_dims(left_eye, axis=0),
        "right_eye": np.expand_dims(right_eye, axis=0),
    }

    if expects_pose:
        pose = estimate_head_pose(landmarks, frame.shape)
        if pose is None:
            print("⚠️ Kopfpose nicht erkennbar")
            continue
        inputs["head_pose"] = np.array([pose], dtype=np.float32)

    prediction = model.predict(inputs, verbose=0)[0]
    x_pred = int(prediction[0] * DISPLAY_W)
    y_pred = int(prediction[1] * DISPLAY_H)

    error_px = np.linalg.norm(np.array([x, y]) - np.array([x_pred, y_pred]))
    error_cm = error_px / DPI * 2.54
    errors.append(error_cm)

    print(f"→ Vorhersage = ({x_pred}, {y_pred}) | Fehler ≈ {error_cm:.2f} cm")

cap.release()
cv2.destroyAllWindows()

# Auswertung
if errors:
    print("\n🎯 Auswertung:")
    print(f"Durchschnittlicher Fehler: {np.mean(errors):.2f} cm")
    print(f"Maximaler Fehler: {np.max(errors):.2f} cm")
else:
    print("❌ Keine gültigen Punkte ausgewertet.")
