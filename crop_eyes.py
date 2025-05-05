import cv2
import mediapipe as mp
import os
import pandas as pd

IMG_DIR = "eye_tracking_data/images"
SAVE_DIR = "eye_tracking_data/eyes"
LABELS_IN = "eye_tracking_data/labels.csv"
LABELS_OUT = "eye_tracking_data/labels_cropped.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Lade Labels
df = pd.read_csv(LABELS_IN)

# Indizes für die äußeren Ecken beider Augen (laut Mediapipe-Dokumentation)
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

new_rows = []

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

    # Koordinaten der Augen
    lx = int((landmarks[LEFT_EYE[0]].x + landmarks[LEFT_EYE[1]].x) / 2 * w)
    ly = int((landmarks[LEFT_EYE[0]].y + landmarks[LEFT_EYE[1]].y) / 2 * h)
    rx = int((landmarks[RIGHT_EYE[0]].x + landmarks[RIGHT_EYE[1]].x) / 2 * w)
    ry = int((landmarks[RIGHT_EYE[0]].y + landmarks[RIGHT_EYE[1]].y) / 2 * h)

    # Crop-Bereich definieren (Augen + Umgebung)
    x_min = min(lx, rx) - 40
    x_max = max(lx, rx) + 40
    y_min = min(ly, ry) - 30
    y_max = max(ly, ry) + 30

    # Bildgrenzen beachten
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)

    eye_crop = image[y_min:y_max, x_min:x_max]

    if eye_crop.size == 0:
        print(f"⚠️ Leerer Crop bei: {filename}")
        continue

    # Speichern
    out_filename = f"eye_{idx:04d}.png"
    out_path = os.path.join(SAVE_DIR, out_filename)
    cv2.imwrite(out_path, eye_crop)

    # Neue Zeile mit gecropptem Bild
    new_rows.append([out_filename, x, y])

# Neue CSV schreiben
pd.DataFrame(new_rows, columns=["filename", "x", "y"]).to_csv(LABELS_OUT, index=False)
print(f"✅ Fertig. Gespeichert in: {SAVE_DIR} und {LABELS_OUT}")
