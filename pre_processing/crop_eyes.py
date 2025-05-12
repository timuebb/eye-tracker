import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
from pre_processing.eye_utils import extract_eye_from_landmarks

# Konfiguration
IMG_DIR = "eye_tracking_data/images"  # ursprüngliche Bilder
SAVE_DIR = "eye_tracking_data/eyes"  # zugeschnittene Augenbilder
LABELS_IN = "eye_tracking_data/labels.csv"
LABELS_OUT = "eye_tracking_data/labels_cropped.csv"
TARGET_SIZE = (64, 64)
SCREEN_W, SCREEN_H = 2560, 1600

os.makedirs(SAVE_DIR, exist_ok=True)

# Mediapipe vorbereiten
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)

# Bestehende Einträge laden
if os.path.exists(LABELS_OUT):
    df_out = pd.read_csv(LABELS_OUT)
    existing_filenames = set(df_out["filename"])
    new_rows = df_out.values.tolist()
else:
    existing_filenames = set()
    new_rows = []

# Input CSV laden
df = pd.read_csv(LABELS_IN)

# Verwende vollständige Augenumrandungen für genaueren Ausschnitt
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

    # Gesichtsausschnitt berechnen
    x_min = int(min(l.x for l in landmarks) * w)
    y_min = int(min(l.y for l in landmarks) * h)
    x_max = int(max(l.x for l in landmarks) * w)
    y_max = int(max(l.y for l in landmarks) * h)
    face_box = (x_min, y_min, x_max - x_min, y_max - y_min)

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
        new_rows.append([out_filename, x_norm, y_norm, eye_label])

# Neue CSV schreiben
pd.DataFrame(new_rows, columns=["filename", "x", "y", "eye"]).to_csv(
    LABELS_OUT, index=False, float_format="%.6f"
)

print(f"✅ Verarbeitung abgeschlossen. Bilder in {SAVE_DIR}, Labels in {LABELS_OUT}")
