import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import pyautogui
from utils.eye_utils import extract_eye_from_landmarks, estimate_head_pose

# Konfiguration
IMG_DIR = "eye_tracking_data/images"  # Rohbilder
SAVE_DIR = "eye_tracking_data/eyes"  # Gespeicherte Augenausschnitte
LABELS_IN = "eye_tracking_data/labels.csv"
LABELS_OUT = "eye_tracking_data/labels_cropped.csv"
TARGET_SIZE = (64, 64)
SCREEN_W, SCREEN_H = pyautogui.size()

os.makedirs(SAVE_DIR, exist_ok=True)

# Mediapipe FaceMesh initialisieren
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)

# Bestehende Ausgabedaten laden
if os.path.exists(LABELS_OUT):
    df_out = pd.read_csv(LABELS_OUT)
    existing_filenames = set(df_out["filename"])
    new_rows = df_out.values.tolist()
else:
    existing_filenames = set()
    new_rows = []

# Input-Labels laden
df = pd.read_csv(LABELS_IN)

# Augenkonturen definieren
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]

for idx, row in df.iterrows():
    filename, x, y = row["filename"], row["x"], row["y"]
    path = os.path.join(IMG_DIR, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"❌ Bild nicht gefunden: {filename}")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print(f"❌ Kein Gesicht erkannt: {filename}")
        continue

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    # Kopfpose berechnen
    pose = estimate_head_pose(landmarks, image.shape)
    if pose is None:
        print(f"⚠️ Kopfpose konnte nicht berechnet werden: {filename}")
        continue

    yaw, pitch, roll = pose

    # Bounding Box des Gesichts (für Extraktion)
    x_min = int(min(l.x for l in landmarks) * w)
    y_min = int(min(l.y for l in landmarks) * h)
    x_max = int(max(l.x for l in landmarks) * w)
    y_max = int(max(l.y for l in landmarks) * h)
    face_box = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Beide Augen extrahieren
    for eye_label, eye_indices in [("left", LEFT_EYE), ("right", RIGHT_EYE)]:
        out_filename = f"{eye_label}_eye_{idx:04d}.png"
        if out_filename in existing_filenames:
            continue

        eye_img = extract_eye_from_landmarks(
            image,
            landmarks,
            eye_indices,
            img_size=TARGET_SIZE,
            face_box=face_box,
            apply_gamma=True,
            gamma_value=1.5,
            apply_hist_eq=True,
            return_raw=True,
        )

        if eye_img is None:
            print(f"⚠️ Kein Auge extrahiert bei {filename} ({eye_label})")
            continue

        cv2.imwrite(os.path.join(SAVE_DIR, out_filename), eye_img)
        x_norm = float(x) / SCREEN_W
        y_norm = float(y) / SCREEN_H
        new_rows.append(
            [
                out_filename,
                round(x_norm, 6),
                round(y_norm, 6),
                eye_label,
                round(yaw, 6),
                round(pitch, 6),
                round(roll, 6),
            ]
        )

# Speichern
pd.DataFrame(
    new_rows, columns=["filename", "x", "y", "eye", "yaw", "pitch", "roll"]
).to_csv(LABELS_OUT, index=False, float_format="%.6f")

print(f"✅ Verarbeitung abgeschlossen. Bilder in {SAVE_DIR}, Labels in {LABELS_OUT}")
