import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, initializers
import matplotlib.pyplot as plt

# Konfiguration
IMG_DIR = "eye_tracking_data/eyes"
LABELS_FILE = "eye_tracking_data/labels_cropped.csv"
IMG_SIZE = (64, 64)
EPOCHS = 50
BATCH_SIZE = 16
NUM_FOLDS = 5
USE_CROSS_VALIDATION = False

# CSV laden & Zielwerte normalisieren
df = pd.read_csv(LABELS_FILE)
df["x"] = df["x"].astype(float)
df["y"] = df["y"].astype(float)

# Paare aus linkem + rechtem Auge bilden
pairs = []
for (_, _), group in df.groupby(["x", "y"]):
    left_rows = group[group["eye"] == "left"]
    right_rows = group[group["eye"] == "right"]
    min_len = min(len(left_rows), len(right_rows))
    for i in range(min_len):
        pairs.append((left_rows.iloc[i], right_rows.iloc[i]))

print(f"👁️ Gefundene Paare: {len(pairs)}")


def load_img(filename):
    path = os.path.join(IMG_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Bild nicht gefunden: {filename}")
        return None
    img = cv2.resize(img, IMG_SIZE)
    return img.astype("float32") / 255.0


# Daten vorbereiten
left_images, right_images, labels = [], [], []
for left_row, right_row in pairs:
    left_img = load_img(left_row["filename"])
    right_img = load_img(right_row["filename"])
    if left_img is None or right_img is None:
        continue
    left_images.append(left_img)
    right_images.append(right_img)
    labels.append([left_row["x"], left_row["y"]])

X_left = np.expand_dims(np.array(left_images), axis=-1)
X_right = np.expand_dims(np.array(right_images), axis=-1)
y = np.array(labels)
X_left, X_right, y = shuffle(X_left, X_right, y, random_state=42)


def augment(x):
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.05)(x)
    return x


def build_model():
    he_init = initializers.HeUniform()

    def cnn_branch(input_tensor):
        x = augment(input_tensor)
        x = tf.keras.layers.GaussianNoise(0.05)(x)
        x = tf.keras.layers.Conv2D(
            32,
            3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
            kernel_initializer=he_init,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Conv2D(
            64,
            3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
            kernel_initializer=he_init,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Conv2D(
            128,
            3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
            kernel_initializer=he_init,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            256,
            3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
            kernel_initializer=he_init,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        return x

    input_left = tf.keras.Input(shape=(64, 64, 1), name="left_eye")
    input_right = tf.keras.Input(shape=(64, 64, 1), name="right_eye")
    encoded_left = cnn_branch(input_left)
    encoded_right = cnn_branch(input_right)

    combined = tf.keras.layers.Concatenate()([encoded_left, encoded_right])
    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer=he_init)(
        combined
    )
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=he_init)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    output = tf.keras.layers.Dense(2, activation="linear")(x)

    model = tf.keras.Model(inputs=[input_left, input_right], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"],
    )
    return model


if USE_CROSS_VALIDATION:
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_left), start=1):
        print(f"\n📂 Fold {fold}/{NUM_FOLDS}")
        Xl_train, Xl_val = X_left[train_idx], X_left[val_idx]
        Xr_train, Xr_val = X_right[train_idx], X_right[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model()
        os.makedirs("models", exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_mae", patience=7, restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                f"models/fold{fold}_dual_model.keras",
                monitor="val_mae",
                save_best_only=True,
                verbose=1,
            ),
            TensorBoard(log_dir=f"logs/fold{fold}_dual"),
        ]

        model.fit(
            {"left_eye": Xl_train, "right_eye": Xr_train},
            y_train,
            validation_data=({"left_eye": Xl_val, "right_eye": Xr_val}, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        preds = model.predict({"left_eye": Xl_val, "right_eye": Xr_val})
        plt.figure(figsize=(8, 8))
        plt.scatter(y_val[:, 0], preds[:, 0], alpha=0.5, label="x")
        plt.scatter(y_val[:, 1], preds[:, 1], alpha=0.5, label="y")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Prediction vs Ground Truth – Fold {fold}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"logs/fold{fold}_scatter.png")

    print(
        "\n✅ Cross-Validation abgeschlossen. Modelle unter models/*.keras gespeichert."
    )
else:
    print("\n🚀 Einzeltraining (kein Fold)")
    Xl_train, Xl_val, Xr_train, Xr_val, y_train, y_val = train_test_split(
        X_left, X_right, y, test_size=0.2, random_state=42
    )

    model = build_model()
    os.makedirs("models", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_mae", patience=7, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            "models/fold1_dual_model.keras",
            monitor="val_mae",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir="logs/fold1_dual"),
    ]

    model.fit(
        {"left_eye": Xl_train, "right_eye": Xr_train},
        y_train,
        validation_data=({"left_eye": Xl_val, "right_eye": Xr_val}, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    preds = model.predict({"left_eye": Xl_val, "right_eye": Xr_val})
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val[:, 0], preds[:, 0], alpha=0.5, label="x")
    plt.scatter(y_val[:, 1], preds[:, 1], alpha=0.5, label="y")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Ground Truth – Einzeltraining")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/single_run_scatter.png")

    print(
        "\n✅ Training abgeschlossen. Modell gespeichert in: models/fold1_dual_model.keras"
    )
