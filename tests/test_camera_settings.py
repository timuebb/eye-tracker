import cv2

cap = cv2.VideoCapture(1)  # oder 1, 2 ausprobieren
if not cap.isOpened():
    print("❌ Kamera nicht verfügbar")
else:
    print("✅ Kamera geöffnet")
    print(
        "Auflösung:",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "x",
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    print("Helligkeit:", cap.get(cv2.CAP_PROP_BRIGHTNESS))
    print("Kontrast:", cap.get(cv2.CAP_PROP_CONTRAST))
    print("Belichtung:", cap.get(cv2.CAP_PROP_EXPOSURE))
    print("Sättigung:", cap.get(cv2.CAP_PROP_SATURATION))
cap.release()
