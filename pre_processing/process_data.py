import os
import cv2
import numpy as np
import pandas as pd
from utils.eye_utils import extract_eye_from_landmarks, estimate_head_pose
import mediapipe as mp

# Konfiguration
RAW_IMG_DIR = "eye_tracking_data/images"
RAW_LABELS_FILE = "eye_tracking_data/labels.csv"
OUTPUT_IMG_DIR = "eye_tracking_data/eyes"
OUTPUT_LABELS_FILE = "eye_tracking_data/labels_cropped.csv"
IMG_SIZE = (64, 64)

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Mediapipe vorbereiten
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]


# Hilfsfunktion zum Laden von Bildern
def load_image(path):
    img = cv2.imread(path)
    return img if img is not None else None


# Bereits verarbeitete Bilddateien aus labels_cropped.csv extrahieren
processed_filenames = set()
if os.path.exists(OUTPUT_LABELS_FILE):
    df_existing = pd.read_csv(OUTPUT_LABELS_FILE)
    processed_filenames = set(
        df_existing["filename"].str.extract(r"eye_(\d+)\.png")[0].dropna()
    )

# Alle Eingabedaten laden
assert os.path.exists(RAW_LABELS_FILE), f"❌ Datei nicht gefunden: {RAW_LABELS_FILE}"
df = pd.read_csv(RAW_LABELS_FILE)

# Nächste Bild-ID basierend auf vorhandenen Dateien ermitteln
existing_output_imgs = [f for f in os.listdir(OUTPUT_IMG_DIR) if f.endswith(".png")]
next_img_id = (
    max(
        [
            int("".join(filter(str.isdigit, f)))
            for f in existing_output_imgs
            if f[:-4].split("_")[-1].isdigit()
        ]
        or [-1]
    )
    + 1
)

# Ausgabe-Datei vorbereiten
file_exists = os.path.isfile(OUTPUT_LABELS_FILE)
mode = "a" if file_exists else "w"
with open(OUTPUT_LABELS_FILE, mode) as out_file:
    if not file_exists:
        out_file.write("filename,x,y,eye,yaw,pitch,roll,tx,ty,tz\n")

    for _, row in df.iterrows():
        fname, rx, ry = row["filename"], row["x"], row["y"]
        base_img_id = os.path.splitext(fname)[0].split("_")[-1]

        if base_img_id in processed_filenames:
            continue  # bereits verarbeitet

        path = os.path.join(RAW_IMG_DIR, fname)
        img = load_image(path)
        if img is None:
            print(f"⚠️ Bild nicht gefunden: {fname}")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print(f"⚠️ Keine Landmarks gefunden in {fname}")
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        pose = estimate_head_pose(landmarks, img.shape)
        if pose is None:
            print(f"⚠️ Kopfpose nicht berechenbar für {fname}")
            continue

        yaw, pitch, roll, tx, ty, tz = pose

        saved = False
        for side, indices in [("left", LEFT_EYE), ("right", RIGHT_EYE)]:
            eye_img = extract_eye_from_landmarks(img, landmarks, indices, IMG_SIZE)
            if eye_img is None:
                continue

            out_fname = f"{side}_eye_{next_img_id:04d}.png"
            cv2.imwrite(
                os.path.join(OUTPUT_IMG_DIR, out_fname), (eye_img * 255).astype("uint8")
            )
            out_file.write(
                f"{out_fname},{rx:.6f},{ry:.6f},{side},{yaw:.6f},{pitch:.6f},{roll:.6f},{tx:.6f},{ty:.6f},{tz:.6f}\n"
            )
            saved = True

        if saved:
            next_img_id += 1

print("✅ Nur neue Bilder wurden verarbeitet. Ergebnisse gespeichert in:")
print(f"  📁 {OUTPUT_IMG_DIR}")
print(f"  📝 {OUTPUT_LABELS_FILE}")
