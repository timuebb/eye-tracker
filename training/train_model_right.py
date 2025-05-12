import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers

# Konfiguration
IMG_DIR = "eye_tracking_data/eyes"
LABELS_FILE = "eye_tracking_data/labels_cropped.csv"
IMG_SIZE = (64, 64)
EPOCHS = 50
BATCH_SIZE = 16
NUM_FOLDS = 5
USE_CROSS_VALIDATION = False  # ← True: K-Fold, False: einmaliges Training

# CSV laden & Zielwerte normalisieren
df = pd.read_csv(LABELS_FILE)
df = df[df["eye"] == "right"].reset_index(drop=True)
df["x"] = df["x"].astype(float)
df["y"] = df["y"].astype(float)

print(f"👁️ Rechte Augen: {len(df)}")

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
    labels.append([row["x"], row["y"]])

X = np.expand_dims(np.array(images), axis=-1)
y = np.array(labels)

X, y = shuffle(X, y, random_state=42)


# Modell definieren
def build_model():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(64, 64, 1)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.GaussianNoise(0.05),
            tf.keras.layers.Conv2D(
                32,
                3,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(
                64,
                3,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(
                128,
                3,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(2, activation="linear"),
        ]
    )


if USE_CROSS_VALIDATION:
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_no = 1

    for train_idx, val_idx in kf.split(X):
        print(f"\n🔄 Fold {fold_no}/{NUM_FOLDS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0005), loss="mse", metrics=["mae"]
        )

        log_dir = f"logs/fold{fold_no}_right"
        checkpoint_path = (
            f"models/fold{fold_no}_eye_tracking_model_right_relative.keras"
        )
        os.makedirs("models", exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_mae", patience=7, restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path, monitor="val_mae", save_best_only=True, verbose=1
            ),
            TensorBoard(log_dir=log_dir),
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        fold_no += 1

    print("\n✅ Cross-Validation abgeschlossen. Modelle in: models/")
else:
    print("\n🚀 Einzeltraining (kein Fold)")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005), loss="mse", metrics=["mae"]
    )

    log_dir = "logs/fold1_right"
    checkpoint_path = "models/fold1_eye_tracking_model_right_relative.keras"
    os.makedirs("models", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_mae", patience=7, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path, monitor="val_mae", save_best_only=True, verbose=1
        ),
        TensorBoard(log_dir=log_dir),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n✅ Training abgeschlossen. Modell gespeichert in: models/")
