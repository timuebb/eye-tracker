import os
import pandas as pd
import numpy as np
import cv2
import pyautogui
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Konfiguration
IMG_DIR = "eye_tracking_data/eyes"
LABELS_FILE = "eye_tracking_data/labels_cropped.csv"
IMG_SIZE = (64, 64)
EPOCHS = 50
BATCH_SIZE = 16
SCREEN_W, SCREEN_H = pyautogui.size()

# CSV laden
df = pd.read_csv(LABELS_FILE)
df = df[df["eye"] == "left"].reset_index(drop=True)
print(f"👁️ Linke Augen: {len(df)}")

# Zielwerte normieren
df["x"] = df["x"].astype(float)
df["y"] = df["y"].astype(float)

# Bilder & Labels laden
images, labels = [], []


def load_img(filename):
    path = os.path.join(IMG_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Bild nicht gefunden: {filename}")
        return None
    img = cv2.resize(img, IMG_SIZE)
    return img.astype("float32") / 255.0


for _, row in df.iterrows():
    img = load_img(row["filename"])
    if img is None:
        continue
    images.append(img)
    labels.append([row["x"], row["y"]])  # schon normiert in CSV

# Zu Arrays
X = np.expand_dims(np.array(images), axis=-1)
y = np.array(labels)

X, y = shuffle(X, y, random_state=42)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Callbacks
early_stop = EarlyStopping(
    monitor="val_mae", patience=7, restore_best_weights=True, verbose=1
)

checkpoint = ModelCheckpoint(
    "eye_tracking_model_left_relative.keras",
    monitor="val_mae",
    save_best_only=True,
    verbose=1,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

# Modell definieren
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(64, 64, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(2, activation="linear"),  # (x, y) normiert
    ]
)

model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# Training
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1,
)

print(
    "✅ Training abgeschlossen. Modell gespeichert als: eye_tracking_model_left_relative.keras"
)
