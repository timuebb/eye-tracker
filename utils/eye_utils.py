import cv2
import numpy as np

# -------------------------------
# Globale Konfiguration
# -------------------------------
DEFAULT_GAMMA = 1.2
CLAHE_CLIP = 1.5
CLAHE_TILE_GRID = (4, 4)
MAX_CROP_SIZE = 100
MIN_CROP_SIZE = 32


def adjust_gamma(image, gamma=DEFAULT_GAMMA):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def enhance_contrast(gray, gamma=DEFAULT_GAMMA, apply_gamma=True, apply_hist_eq=True):
    if apply_gamma:
        gray = adjust_gamma(gray, gamma)
    if apply_hist_eq:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE_GRID)
        gray = clahe.apply(gray)
    return gray


def extract_eye_from_landmarks(
    frame,
    landmarks,
    eye_indices,
    img_size=(64, 64),
    face_box=None,
    apply_gamma=True,
    gamma_value=DEFAULT_GAMMA,
    apply_hist_eq=True,
    return_raw=False,
):
    h, w, _ = frame.shape

    if face_box is not None:
        x_min, y_min, box_w, box_h = face_box
        crop = frame[y_min : y_min + box_h, x_min : x_min + box_w]
        if crop.size == 0:
            return None
        coords = np.array(
            [
                (int(landmarks[i].x * w - x_min), int(landmarks[i].y * h - y_min))
                for i in eye_indices
            ]
        )
        ref_w, ref_h = box_w, box_h
    else:
        crop = frame
        coords = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
        )
        ref_w, ref_h = w, h

    if len(coords) < 2:
        return None

    x_min_eye = np.min(coords[:, 0])
    x_max_eye = np.max(coords[:, 0])
    y_min_eye = np.min(coords[:, 1])
    y_max_eye = np.max(coords[:, 1])

    cx = (x_min_eye + x_max_eye) // 2
    cy = (y_min_eye + y_max_eye) // 2
    eye_w = x_max_eye - x_min_eye
    eye_h = y_max_eye - y_min_eye

    crop_size = int(max(MIN_CROP_SIZE, min(max(eye_w, eye_h) * 2.2, MAX_CROP_SIZE)))

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

    return resized if return_raw else np.expand_dims(normalized, axis=-1)


def estimate_head_pose(landmarks, image_shape):
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [-30.0, -30.0, -30.0],
            [30.0, -30.0, -30.0],
            [-30.0, 30.0, -30.0],
            [30.0, 30.0, -30.0],
            [0.0, 75.0, -30.0],
        ]
    )

    h, w = image_shape[:2]
    image_points = np.array(
        [
            [landmarks[1].x * w, landmarks[1].y * h],
            [landmarks[33].x * w, landmarks[33].y * h],
            [landmarks[263].x * w, landmarks[263].y * h],
            [landmarks[61].x * w, landmarks[61].y * h],
            [landmarks[291].x * w, landmarks[291].y * h],
            [landmarks[199].x * w, landmarks[199].y * h],
        ],
        dtype="double",
    )

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    pitch = np.arctan2(-rotation_matrix[2, 0], sy)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    yaw, pitch, roll = np.degrees([yaw, pitch, roll])
    x, y, z = translation_vector.flatten()

    return np.array([yaw, pitch, roll, x, y, z])
