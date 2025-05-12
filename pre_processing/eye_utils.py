import cv2
import numpy as np


def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def enhance_contrast(gray, gamma=1.5, apply_gamma=True, apply_hist_eq=True):
    if apply_gamma:
        gray = adjust_gamma(gray, gamma)
    if apply_hist_eq:
        gray = cv2.equalizeHist(gray)
    return gray


def extract_eye_from_landmarks(
    frame,
    landmarks,
    eye_indices,
    img_size=(64, 64),
    face_box=None,
    apply_gamma=True,
    gamma_value=1.5,
    apply_hist_eq=True,
    return_raw=False,
):
    h, w, _ = frame.shape

    # Region beschränken, falls face_box vorhanden
    if face_box is not None:
        x_min, y_min, box_w, box_h = face_box
        crop = frame[y_min : y_min + box_h, x_min : x_min + box_w]
        if crop.size == 0:
            return None
        coords = [
            (int(landmarks[i].x * w - x_min), int(landmarks[i].y * h - y_min))
            for i in eye_indices
        ]
        ref_w, ref_h = box_w, box_h
    else:
        crop = frame
        coords = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices
        ]
        ref_w, ref_h = w, h

    if len(coords) < 2:
        return None

    coords = np.array(coords)
    cx, cy = np.mean(coords, axis=0).astype(int)
    eye_width = int(np.max(np.linalg.norm(coords - [cx, cy], axis=1)) * 2)

    crop_size = int(max(32, min(eye_width * 2, 100)))
    x_start = max(cx - crop_size // 2, 0)
    y_start = max(cy - crop_size // 2, 0)
    x_end = min(cx + crop_size // 2, ref_w)
    y_end = min(cy + crop_size // 2, ref_h)

    eye_img = crop[y_start:y_end, x_start:x_end]
    if eye_img.size == 0:
        return None

    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_contrast(
        gray, gamma=gamma_value, apply_gamma=apply_gamma, apply_hist_eq=apply_hist_eq
    )

    resized = cv2.resize(enhanced, img_size)
    normalized = resized.astype("float32") / 255.0

    if return_raw:
        return resized
    return np.expand_dims(normalized, axis=-1)  # Shape: (64, 64, 1)


def estimate_head_pose(landmarks, image_shape):
    # 3D-Modellpunkte im Gesicht (in mm, grobe Annahmen)
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],  # Nase
            [-30.0, -30.0, -30.0],  # Linkes Auge
            [30.0, -30.0, -30.0],  # Rechtes Auge
            [-30.0, 30.0, -30.0],  # Linker Mundwinkel
            [30.0, 30.0, -30.0],  # Rechter Mundwinkel
            [0.0, 75.0, -30.0],  # Kinn
        ]
    )

    h, w = image_shape[:2]
    image_points = np.array(
        [
            [landmarks[1].x * w, landmarks[1].y * h],  # Nase
            [landmarks[33].x * w, landmarks[33].y * h],  # Linkes Auge
            [landmarks[263].x * w, landmarks[263].y * h],  # Rechtes Auge
            [landmarks[61].x * w, landmarks[61].y * h],  # Mund links
            [landmarks[291].x * w, landmarks[291].y * h],  # Mund rechts
            [landmarks[199].x * w, landmarks[199].y * h],  # Kinn
        ],
        dtype="double",
    )

    # Kamera-Matrix (angenommene Werte)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))  # keine Verzerrung
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return None

    # Rotation in Euler-Winkel (Roll, Pitch, Yaw)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    pitch = np.arctan2(-rotation_matrix[2, 0], sy)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    return np.degrees([yaw, pitch, roll])  # in Grad
