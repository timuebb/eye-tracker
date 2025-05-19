import cv2
import numpy as np
import os
import time
import pyautogui
import mediapipe as mp
from utils.eye_utils import estimate_head_pose

# Konfiguration
SAVE_DIR = "eye_tracking_data"
IMG_DIR = os.path.join(SAVE_DIR, "images")
LABEL_FILE = os.path.join(SAVE_DIR, "labels.csv")
os.makedirs(IMG_DIR, exist_ok=True)

cap = cv2.VideoCapture(1)
screen_w, screen_h = pyautogui.size()
grid_size = 5
IMG_EXT = ".png"

# Raster erzeugen
raster_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
center = (grid_size - 1) / 2

# Samples pro Zelle je nach Abstand zur Mitte
samples_per_cell_map = {
    (i, j): int(
        10 + 40 * np.linalg.norm([(i - center) / center, (j - center) / center])
    )
    for i, j in raster_cells
}

# Shuffle Reihenfolge für zufällige Anzeige
np.random.seed(42)
np.random.shuffle(raster_cells)

cell_w = 1.0 / grid_size
cell_h = 1.0 / grid_size

# Bestehende Bilder analysieren
existing_images = [f for f in os.listdir(IMG_DIR) if f.endswith(IMG_EXT)]
img_id = (
    max(
        [
            int("".join(filter(str.isdigit, f)))
            for f in existing_images
            if f[: -len(IMG_EXT)].split("_")[-1].isdigit()
        ]
        or [-1]
    )
    + 1
)
start_id = img_id  # Merken, ab wo neue Bilder beginnen

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

cv2.namedWindow("target", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("target", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

file_exists = os.path.isfile(LABEL_FILE)
mode = "a" if file_exists else "w"

try:
    with open(LABEL_FILE, mode) as f:
        if not file_exists:
            f.write("filename,x,y,yaw,pitch,roll\n")

        for i, j in raster_cells:
            rx = (i + np.random.uniform(0.2, 0.8)) * cell_w
            ry = (j + np.random.uniform(0.2, 0.8)) * cell_h
            x = int(screen_w * rx)
            y = int(screen_h * ry)
            samples = samples_per_cell_map[(i, j)]

            print(f"👉 Schaue auf Raster ({i},{j}) ({x}, {y}) | {samples} Bilder")

            # Countdown anzeigen
            for countdown in [3, 2, 1]:
                screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv2.circle(screen, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(
                    screen,
                    str(countdown),
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (255, 255, 255),
                    10,
                )
                cv2.imshow("target", screen)
                cv2.waitKey(1000)

            for _ in range(samples):
                screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv2.circle(screen, (x, y), 10, (0, 0, 255), -1)
                cv2.imshow("target", screen)
                cv2.waitKey(1)

                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Kamera-Fehler")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    pose = estimate_head_pose(landmarks, frame.shape)
                    if pose is None:
                        print("⚠️ Kopfpose nicht berechenbar")
                        continue

                    yaw, pitch, roll = pose
                    fname = f"eye_{img_id:04d}{IMG_EXT}"
                    cv2.imwrite(os.path.join(IMG_DIR, fname), frame)
                    f.write(
                        f"{fname},{rx:.6f},{ry:.6f},{yaw:.6f},{pitch:.6f},{roll:.6f}\n"
                    )
                    img_id += 1

                time.sleep(0.12)  # leicht verkürzt

except KeyboardInterrupt:
    print("\n⏹️ Aufnahme manuell gestoppt. Bisherige Daten wurden gespeichert.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    end_id = img_id - 1
    if end_id >= start_id:
        print(
            f"\n✅ Neue Bilder gespeichert von eye_{start_id:04d}{IMG_EXT} bis eye_{end_id:04d}{IMG_EXT}"
        )
    else:
        print("ℹ️ Keine neuen Bilder aufgenommen.")
