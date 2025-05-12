import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Modell laden (Single-Eye-Modell, z. B. eye_tracking_model_left.keras)
model = tf.keras.models.load_model("eye_tracking_model_left_relative.keras")

# Mediapipe vorbereiten
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Kamera starten
cap = cv2.VideoCapture(1)  # ggf. 0 oder 2 ausprobieren

# Nur linkes Auge
LEFT_EYE = [33, 133]
IMG_SIZE = (64, 64)

# Anzeigegröße entspricht direkt der Trainingsgröße
DISPLAY_W, DISPLAY_H = 2560, 1600
WINDOW_NAME = "Vorhergesagter Blickpunkt (linkes Auge)"


def extract_eye(image, landmarks, eye_indices):
    h, w, _ = image.shape
    p1, p2 = landmarks[eye_indices[0]], landmarks[eye_indices[1]]
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    eye_width = int(np.hypot(x2 - x1, y2 - y1))
    crop_size = int(max(32, min(eye_width * 1.5, 100)))
    x_start = max(cx - crop_size // 2, 0)
    y_start = max(cy - crop_size // 2, 0)
    x_end = min(cx + crop_size // 2, w)
    y_end = min(cy + crop_size // 2, h)
    eye_img = image[y_start:y_end, x_start:x_end]
    if eye_img.size == 0:
        return None
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    return resized  # uint8, 0–255


while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kein Kamerabild")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = extract_eye(frame, landmarks, LEFT_EYE)

        if left_eye is not None:
            # Normalisierung NUR HIER wie beim Training
            left_eye_input = left_eye.astype("float32") / 255.0
            left_eye_input = np.expand_dims(
                left_eye_input, axis=(0, -1)
            )  # (1, 64, 64, 1)

            prediction = model.predict(left_eye_input, verbose=0)[0]

            # 🔍 Debug: Normierte Koordinaten anzeigen
            print(f"Vorhersage (x_norm, y_norm): {prediction}")

            # 🔍 Debug: Auge anzeigen, das ins Modell eingeht
            eye_preview = (left_eye[0] * 255).astype("uint8")
            cv2.imshow("Eye Input", cv2.resize(eye_preview, (128, 128)))

            # Koordinaten auf Bildschirmgröße skalieren
            x_pred = int(prediction[0] * DISPLAY_W)
            y_pred = int(prediction[1] * DISPLAY_H)

            # Ergebnis anzeigen
            display = np.ones((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8) * 255
            cv2.circle(display, (x_pred, y_pred), 20, (0, 0, 255), -1)
            cv2.imshow(WINDOW_NAME, display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
