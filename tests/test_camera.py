import cv2

print("Suche verfügbare Kameras...")
for index in range(5):
    cap = cv2.VideoCapture(index)
    ret, frame = cap.read()
    if ret:
        print(f"✅ Kamera {index} funktioniert.")
        cv2.imshow(f"Kamera {index}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"❌ Kamera {index} nicht verfügbar.")
    cap.release()
