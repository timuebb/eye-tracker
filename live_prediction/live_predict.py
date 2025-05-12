import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from utils.eye_utils import extract_eye_from_landmarks

# Konfiguration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dual = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "fold1_dual_model.keras")
)
model_left = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "eye_tracking_model_left_relative.keras")
)
model_right = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "eye_tracking_model_right_relative.keras")
)

# Mediapipe vorbereiten
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

cap = cv2.VideoCapture(1)

DISPLAY_W, DISPLAY_H = pyautogui.size()
WINDOW_NAME = "Live-Prediction: dynamisches Auge"

# Vollständige Augenumrandung für robusteren Zuschnitt
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382]
IMG_SIZE = (64, 64)

# Optional: letzte Vorhersage zur Glättung
last_prediction = None


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
    ret, frame = cap.read()
    if not ret:
        print("❌ Kein Kamerabild")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        face_box = get_face_box(landmarks, frame.shape)

        # Augen extrahieren (auf Gesichtsausschnitt begrenzt)
        left_eye = extract_eye_from_landmarks(
            frame, landmarks, LEFT_EYE, IMG_SIZE, face_box
        )
        right_eye = extract_eye_from_landmarks(
            frame, landmarks, RIGHT_EYE, IMG_SIZE, face_box
        )

        predictions = []

        if left_eye is not None:
            input_left = np.expand_dims(left_eye, axis=0)
            pred_left = model_left.predict(input_left, verbose=0)[0]
            predictions.append(("left", pred_left))

        if right_eye is not None:
            input_right = np.expand_dims(right_eye, axis=0)
            pred_right = model_right.predict(input_right, verbose=0)[0]
            predictions.append(("right", pred_right))

        if left_eye is not None and right_eye is not None:
            input_left = np.expand_dims(left_eye, axis=0)
            input_right = np.expand_dims(right_eye, axis=0)
            pred_dual = model_dual.predict(
                {"left_eye": input_left, "right_eye": input_right}, verbose=0
            )[0]
            predictions.append(("dual", pred_dual))

        if predictions:
            # Bestes Modell nach Distanz zur letzten Vorhersage auswählen
            best_name, best_pred = predictions[0]
            if last_prediction is not None:
                min_dist = np.inf
                for name, pred in predictions:
                    dist = np.linalg.norm(np.array(pred) - last_prediction)
                    if dist < min_dist:
                        best_name, best_pred = name, pred
                        min_dist = dist
            last_prediction = best_pred

            x_scaled = int(best_pred[0] * DISPLAY_W)
            y_scaled = int(best_pred[1] * DISPLAY_H)

            display = np.ones((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8) * 255
            cv2.circle(display, (x_scaled, y_scaled), 20, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"Modell: {best_name}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                4,
            )
            cv2.imshow(WINDOW_NAME, display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
