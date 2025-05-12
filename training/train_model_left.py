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
FOLDS = 5
USE_CROSS_VALIDATION = (
    False  # <<< HIER ANPASSEN: True = KFold, False = train_test_split
)

# Daten einlesen und filtern
all_data = pd.read_csv(LABELS_FILE)
df = all_data[all_data["eye"] == "left"].reset_index(drop=True)

# Bilder und Labels laden
images, labels = [], []


def load_img(filename):
    path = os.path.join(IMG_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
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
def build_eye_model():
    inputs = tf.keras.Input(shape=(64, 64, 1))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.05)(x)
    x = tf.keras.layers.RandomTranslation(0.05, 0.05)(x)
    x = tf.keras.layers.GaussianNoise(0.05)(x)

    x = tf.keras.layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    return tf.keras.Model(inputs, outputs)


os.makedirs("models", exist_ok=True)

if USE_CROSS_VALIDATION:
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n\n📂 Fold {fold+1}/{FOLDS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_eye_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0005), loss="mse", metrics=["mae"]
        )

        callbacks = [
            EarlyStopping(
                monitor="val_mae", patience=7, restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                f"models/fold{fold+1}_eye_tracking_model_left_relative.keras",
                monitor="val_mae",
                save_best_only=True,
                verbose=1,
            ),
            TensorBoard(log_dir=f"logs/fold{fold+1}_left"),
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

    print("\n✅ Cross-Validation abgeschlossen. Modelle in 'models/' gespeichert.")

else:
    print("\n📂 Einfaches Training ohne Cross-Validation")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_eye_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005), loss="mse", metrics=["mae"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_mae", patience=7, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            "models/eye_tracking_model_left_relative.keras",
            monitor="val_mae",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir="logs/single_run_left"),
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

    print("\n✅ Training abgeschlossen. Modell gespeichert in 'models/'.")
