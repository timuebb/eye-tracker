import pandas as pd
import pyautogui

# Original- und Zielauflösung
OLD_W, OLD_H = 2560, 1600
NEW_W, NEW_H = pyautogui.size()


# Pfade
LABELS_IN = "eye_tracking_data/labels.csv"
LABELS_OUT = "eye_tracking_data/labels_scaled.csv"

# CSV laden
df = pd.read_csv(LABELS_IN)

# Skalieren
df["x"] = (df["x"] * NEW_W / OLD_W).round().astype(int)
df["y"] = (df["y"] * NEW_H / OLD_H).round().astype(int)  ##shit gerundet

# Speichern
df.to_csv(LABELS_OUT, index=False)
print(f"✅ Neue Labels gespeichert unter: {LABELS_OUT}")
