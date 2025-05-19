import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.eye_utils import enhance_contrast

# Konfiguration
BASE_DIR_CSV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(BASE_DIR_CSV, "eye_tracking_data", "labels_cropped.csv")
BASE_DIR_IMAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR_IMAGE, "eye_tracking_data", "eyes")
IMG_SIZE = (64, 64)

# CSV laden
assert os.path.exists(LABELS_CSV), f"❌ Datei nicht gefunden: {LABELS_CSV}"
df = pd.read_csv(LABELS_CSV)

# Fehlerzählung
missing_files = 0
invalid_coords = 0
loaded_images = []

for idx, row in df.iterrows():
    filename = row["filename"]
    x, y = row["x"], row["y"]
    img_path = os.path.join(IMG_DIR, filename)

    if not os.path.isfile(img_path):
        print(f"❌ Bild fehlt: {filename}")
        missing_files += 1
        continue

    if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
        print(f"⚠️ Ungültige Koordinaten in {filename}: x={x}, y={y}")
        invalid_coords += 1

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Konnte nicht geladen werden: {filename}")
        continue

    img = cv2.resize(img, IMG_SIZE)
    enhanced = enhance_contrast(img)
    loaded_images.append((img, enhanced, x, y))

print("\n✅ Test abgeschlossen.")
print(f"🔍 Bilder überprüft: {len(df)}")
print(f"🧩 Fehlende Dateien: {missing_files}")
print(f"📐 Ungültige x/y-Koordinaten: {invalid_coords}")

# Beispielbilder anzeigen
if loaded_images:
    n = min(5, len(loaded_images))
    examples = random.sample(loaded_images, n)

    for i, (original, processed, x, y) in enumerate(examples, 1):
        cv2.imshow(f"{i}. Original ({x:.2f}, {y:.2f})", original)
        cv2.imshow(f"{i}. Kontrastverstärkt", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Streudiagramm der Koordinaten
plt.figure(figsize=(6, 6))
plt.scatter(df["x"], df["y"], alpha=0.4, c="blue")
plt.title("📍 Verteilung der Blickpunkte")
plt.xlabel("x (normiert)")
plt.ylabel("y (normiert)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.gca()
plt.show()
