import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Pfade
IMG_DIR = "eye_tracking_data/eyes"
LABELS_FILE = "eye_tracking_data/labels_cropped.csv"

# Trainingsparameter
IMG_SIZE = (64, 64)  # Zielgröße für Augenbilder
EPOCHS = 10
BATCH_SIZE = 16

# Labels laden
df = pd.read_csv(LABELS_FILE)

# Bilder + Labels vorbereiten
images = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, row["filename"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0  # Normalisieren
    images.append(img)
    labels.append([row["x"], row["y"]])  # Bildschirm-Koordinaten

images = np.expand_dims(np.array(images), axis=-1)  # (N, 64, 64, 1)
labels = np.array(labels)

# Trainings-/Validierungsdaten splitten
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Modell definieren
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2),  # X und Y Koordinate
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Training
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# Speichern
model.save("eye_tracking_model.h5")
print("✅ Modell gespeichert als eye_tracking_model.h5")
