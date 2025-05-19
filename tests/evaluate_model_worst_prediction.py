import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import mediapipe as mp
from utils.eye_utils import extract_eye_from_landmarks, estimate_head_pose

# Konfiguration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fold1_dual_model.keras")
LABELS_CSV = os.path.join(BASE_DIR, "eye_tracking_data", "labels.csv")
IMG_DIR = os.path.join(BASE_DIR, "eye_tracking_data", "images")
IMG_SIZE = (64, 64)
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]
DPI = 110  # Bildschirm DPI für Umrechnung Pixel -> cm
USE_HEAD_POSE = False  # True aktivieren, wenn Kopfpose genutzt werden soll

# Bildschirmauflösung (ggf. mit pyautogui.size() dynamisch ermitteln)
screen_w, screen_h = 1440, 900
cm_per_pixel = 2.54 / DPI

# Modell laden
model = tf.keras.models.load_model(MODEL_PATH)
expects_pose = "head_pose" in [inp.name.split(":")[0] for inp in model.inputs]

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def get_landmarks_from_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None


# Daten laden
df = pd.read_csv(LABELS_CSV)
errors = []
norm_mae = []

print(f"Modell erwartet Kopfpose: {expects_pose} | aktiviert: {USE_HEAD_POSE}")

for _, row in df.iterrows():
    filename = row["filename"]
    true_x_norm = row["x"]
    true_y_norm = row["y"]
    true_x = true_x_norm * screen_w
    true_y = true_y_norm * screen_h

    pose = [row["yaw"], row["pitch"], row["roll"]] if USE_HEAD_POSE else None

    img_path = os.path.join(IMG_DIR, filename)
    image = cv2.imread(img_path)
    if image is None:
        continue

    landmarks = get_landmarks_from_image(image)
    if landmarks is None:
        continue

    left_eye = extract_eye_from_landmarks(image, landmarks, LEFT_EYE, IMG_SIZE)
    right_eye = extract_eye_from_landmarks(image, landmarks, RIGHT_EYE, IMG_SIZE)

    if left_eye is None or right_eye is None:
        continue

    inputs = {
        "left_eye": np.expand_dims(left_eye, axis=0),
        "right_eye": np.expand_dims(right_eye, axis=0),
    }
    if USE_HEAD_POSE and pose:
        inputs["head_pose"] = np.array([pose], dtype=np.float32)

    pred = model.predict(inputs, verbose=0)[0]
    pred_x = pred[0] * screen_w
    pred_y = pred[1] * screen_h

    pixel_error = np.linalg.norm([true_x - pred_x, true_y - pred_y])
    cm_error = pixel_error * cm_per_pixel
    errors.append((filename, cm_error, (true_x, true_y), (pred_x, pred_y)))

    # MAE berechnen auf normierten Werten (0–1)
    norm_mae.append(abs(true_x_norm - pred[0]) + abs(true_y_norm - pred[1]))

# Top 50 größte Fehler
errors.sort(key=lambda e: -e[1])
top_50 = errors[:50]

print("\n📉 Top 50 Fehler (in cm):")
for fname, err, true, pred in top_50:
    print(f"{fname}: Fehler = {err:.2f} cm | True = {true} | Vorhersage = {pred}")

mean_error = np.mean([e[1] for e in errors])
max_error = np.max([e[1] for e in errors])
val_mae = np.mean(norm_mae)

print(
    f"\n🎯 Durchschnittlicher Fehler: {mean_error:.2f} cm | Maximaler Fehler: {max_error:.2f} cm"
)
print(f"📏 Val MAE (normiert): {val_mae:.4f}")
