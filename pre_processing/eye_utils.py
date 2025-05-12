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
