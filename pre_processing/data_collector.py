import cv2
import numpy as np
import os
import time
import mediapipe as mp
from pre_processing.eye_utils import extract_eye_from_landmarks, estimate_head_pose

SAVE_DIR = "eye_tracking_data"
IMG_DIR = os.path.join(SAVE_DIR, "eyes")
LABEL_FILE = os.path.join(SAVE_DIR, "labels_cropped.csv")
os.makedirs(IMG_DIR, exist_ok=True)

cap = cv2.VideoCapture(1)
screen_w, screen_h = 2560, 1600
samples_per_point = 30
IMG_SIZE = (64, 64)
steps = np.linspace(0.05, 0.95, 5)
raster = [(x, y) for y in steps for x in steps]

existing_images = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
img_id = (
    max(
        [
            int("".join(filter(str.isdigit, f)))
            for f in existing_images
            if f[:-4].split("_")[-1].isdigit()
        ]
        or [-1]
    )
    + 1
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]


def draw_screen(x, y, countdown):
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


cv2.namedWindow("target", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("target", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

file_exists = os.path.isfile(LABEL_FILE)
mode = "a" if file_exists else "w"
with open(LABEL_FILE, mode) as f:
    if not file_exists:
        f.write("filename,x,y,eye,yaw,pitch,roll\n")

    for rx, ry in raster:
        x = int(screen_w * rx)
        y = int(screen_h * ry)
        print(f"👉 Schaue auf Punkt ({x}, {y})")

        for countdown in [3, 2, 1]:
            draw_screen(x, y, countdown)
            cv2.waitKey(1000)

        for _ in range(samples_per_point):
            draw_screen(x, y, "●")
            cv2.waitKey(1)

            ret, frame = cap.read()
            if not ret:
                print("⚠️ Kamera-Fehler")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                head_pose = estimate_head_pose(landmarks, frame.shape)
                if head_pose is None:
                    print("⚠️ Kopfpose nicht berechenbar")
                    continue
                yaw, pitch, roll = head_pose

                left = extract_eye_from_landmarks(frame, landmarks, LEFT_EYE, IMG_SIZE)
                right = extract_eye_from_landmarks(
                    frame, landmarks, RIGHT_EYE, IMG_SIZE
                )

                if left is not None:
                    fname = f"left_eye_{img_id:04d}.png"
                    cv2.imwrite(
                        os.path.join(IMG_DIR, fname), (left * 255).astype("uint8")
                    )
                    f.write(
                        f"{fname},{rx},{ry},left,{yaw:.2f},{pitch:.2f},{roll:.2f}\n"
                    )

                if right is not None:
                    fname = f"right_eye_{img_id:04d}.png"
                    cv2.imwrite(
                        os.path.join(IMG_DIR, fname), (right * 255).astype("uint8")
                    )
                    f.write(
                        f"{fname},{rx},{ry},right,{yaw:.2f},{pitch:.2f},{roll:.2f}\n"
                    )

                img_id += 1

            cv2.waitKey(1)
            time.sleep(0.15)

cap.release()
cv2.destroyAllWindows()
print("✅ Kalibrierung abgeschlossen.")
