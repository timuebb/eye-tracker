import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from datetime import datetime
import matplotlib.pyplot as plt
from utils.camery_utils import find_internal_camera
from utils.eye_utils import extract_eye_from_landmarks, estimate_head_pose

# Konfiguration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fold1_dual_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)
expects_pose = "head_pose" in [inp.name.split(":")[0] for inp in model.inputs]

cap = cv2.VideoCapture(1)


def get_display_size():
    return pyautogui.size()


WINDOW_NAME = "Live-Prediction: Dual-Modell"
IMG_SIZE = (64, 64)
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]

# Logging vorbereiten
LOG_FILE = os.path.join(
    BASE_DIR, "logs", f"live_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
log_file = open(LOG_FILE, "w")

predictions_log = []


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

log(f"Modell geladen: {MODEL_PATH}")
log(f"Kopfpose erwartet: {expects_pose}")

last_prediction = None
prev_display_size = None


def get_face_box(landmarks, image_shape):
    h, w, _ = image_shape
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min = int(np.min(xs))
    y_min = int(np.min(ys))
    x_max = int(np.max(xs))
    y_max = int(np.max(ys))
    return x_min, y_min, x_max - x_min, y_max - y_min


while True:
    DISPLAY_W, DISPLAY_H = get_display_size()
    if (DISPLAY_W, DISPLAY_H) != prev_display_size:
        log(f"Neue Displaygröße erkannt: {DISPLAY_W}x{DISPLAY_H}")
        prev_display_size = (DISPLAY_W, DISPLAY_H)

    ret, frame = cap.read()
    if not ret:
        log("❌ Kein Kamerabild")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        face_box = get_face_box(landmarks, frame.shape)

        left_eye = extract_eye_from_landmarks(
            frame, landmarks, LEFT_EYE, IMG_SIZE, face_box=face_box
        )
        right_eye = extract_eye_from_landmarks(
            frame, landmarks, RIGHT_EYE, IMG_SIZE, face_box=face_box
        )

        if left_eye is not None and right_eye is not None:
            input_left = np.expand_dims(left_eye, axis=0)
            input_right = np.expand_dims(right_eye, axis=0)
            inputs = {"left_eye": input_left, "right_eye": input_right}

            pose_info = ""
            if expects_pose:
                pose = estimate_head_pose(landmarks, frame.shape)
                if pose is None:
                    log("⚠️ Kopfpose nicht bestimmbar – Frame übersprungen")
                    continue

                # pose = [yaw, pitch, roll, tx, ty, tz]
                inputs["head_pose"] = np.expand_dims(pose.astype("float32"), axis=0)
                yaw, pitch, roll, tx, ty, tz = pose

                pose_info = (
                    f"Yaw={yaw:.1f}°, Pitch={pitch:.1f}°, Roll={roll:.1f}° | "
                    f"Tx={tx:.1f}, Ty={ty:.1f}, Tz={tz:.1f}"
                )
                log(f"🔍 DEBUG: head_pose + trans = {pose}")

            prediction = model.predict(inputs, verbose=0)[0]

            if last_prediction is not None:
                prediction = 0.6 * prediction + 0.4 * last_prediction
            last_prediction = prediction

            x_norm, y_norm = prediction
            # 🔍 DEBUG: Bereich prüfen
            if not (0 <= x_norm <= 1) or not (0 <= y_norm <= 1):
                log(
                    f"⚠️ WARNUNG: Normierte Koordinaten außerhalb von [0,1]: ({x_norm:.3f}, {y_norm:.3f})"
                )

            x_scaled = int(x_norm * DISPLAY_W)
            y_scaled = int(y_norm * DISPLAY_H)
            y_inverted = int((1 - y_norm) * DISPLAY_H)

            log(f"🔍 DEBUG: normiert = ({x_norm:.3f}, {y_norm:.3f})")
            log(f"           x_scaled = {x_scaled}, y_scaled = {y_scaled}")
            log(f"       y_inverted_scaled = {y_inverted}")

            predictions_log.append([x_norm, y_norm])

            display = np.ones((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8) * 255
            cv2.circle(display, (x_scaled, y_scaled), 20, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"Dual-Modell{' + Pose' if expects_pose else ''}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                4,
            )
            if expects_pose:
                cv2.putText(
                    display,
                    pose_info,
                    (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    3,
                )

            cv2.imshow(WINDOW_NAME, display)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
