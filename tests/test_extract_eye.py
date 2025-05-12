import cv2
import mediapipe as mp
import numpy as np
from pre_processing.eye_utils import extract_eye_from_landmarks

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

# Kamera starten
cap = cv2.VideoCapture(1)  # ggf. 1, wenn du mehrere Kameras hast

# Auge-Landmark-Indizes
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
IMG_SIZE = (64, 64)


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

        left_eye = extract_eye_from_landmarks(
            frame, landmarks, LEFT_EYE, IMG_SIZE, face_box
        )
        right_eye = extract_eye_from_landmarks(
            frame, landmarks, RIGHT_EYE, IMG_SIZE, face_box
        )

        if left_eye is not None:
            eye_img = (left_eye * 255).astype("uint8")
            cv2.imshow("Left Eye", cv2.resize(eye_img, (128, 128)))

        if right_eye is not None:
            eye_img = (right_eye * 255).astype("uint8")
            cv2.imshow("Right Eye", cv2.resize(eye_img, (128, 128)))

    if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
        break

cap.release()
cv2.destroyAllWindows()
