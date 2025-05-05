import cv2
import numpy as np
import os
import random
import time

# Pfade
SAVE_DIR = "eye_tracking_data"
IMG_DIR = os.path.join(SAVE_DIR, "images")
LABEL_FILE = os.path.join(SAVE_DIR, "labels.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# Webcam starten
cap = cv2.VideoCapture(0)
screen_w, screen_h = 1920, 1080  # Bildschirmauflösung (anpassen!)
num_samples = 100  # Anzahl der Datenpunkte

print("Starte Datensammlung in 3 Sekunden...")
time.sleep(3)

with open(LABEL_FILE, "w") as f:
    f.write("filename,x,y\n")

    for i in range(num_samples):
        # Zufälliger Punkt
        x = random.randint(100, screen_w - 100)
        y = random.randint(100, screen_h - 100)

        # Schwarzes Bild mit rotem Punkt (als Target)
        screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(screen, (x, y), 20, (0, 0, 255), -1)

        # Zeige für kurze Zeit den Punkt
        cv2.namedWindow("target", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("target", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("target", screen)
        cv2.waitKey(1000)

        # Bild aufnehmen
        ret, frame = cap.read()
        if not ret:
            print("Fehler bei Webcam.")
            continue

        # Bild speichern
        filename = f"img_{i:04d}.png"
        filepath = os.path.join(IMG_DIR, filename)
        cv2.imwrite(filepath, frame)

        # Label speichern
        f.write(f"{filename},{x},{y}\n")

        # Kurze Pause
        cv2.destroyWindow("target")
        time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
print("Datensammlung abgeschlossen.")
