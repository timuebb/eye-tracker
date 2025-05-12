import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
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
FOLDS = 5
USE_CROSS_VALIDATION = False
USE_HEAD_POSE = True  # ← Aktiviere bei vorhandenen yaw/pitch/roll-Spalten

# Daten einlesen und filtern
df = pd.read_csv(LABELS_FILE)
df = df[df["eye"] == "left"].reset_index(drop=True)
df["x"] = df["x"].astype(float)
df["y"] = df["y"].astype(float)

if USE_HEAD_POSE and all(col in df.columns for col in ["yaw", "pitch", "roll"]):
    df["yaw"] = df["yaw"].astype(float)
    df["pitch"] = df["pitch"].astype(float)
    df["roll"] = df["roll"].astype(float)
    use_pose = True
else:
    use_pose = False

# Bild- und Label-Arrays
images, labels, pose_data = [], [], []


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
    if use_pose:
        pose_data.append([row["yaw"], row["pitch"], row["roll"]])

X = np.expand_dims(np.array(images), axis=-1)
y = np.array(labels)

if use_pose:
    X_pose = np.array(pose_data)
    X, X_pose, y = shuffle(X, X_pose, y, random_state=42)
else:
    X, y = shuffle(X, y, random_state=42)


# Modell definieren
def build_eye_model():
    he_init = initializers.HeUniform()

    input_eye = tf.keras.Input(shape=(64, 64, 1), name="eye_input")
    x = tf.keras.layers.RandomRotation(0.05)(input_eye)
    x = tf.keras.layers.RandomZoom(0.05)(x)
    x = tf.keras.layers.GaussianNoise(0.05)(x)

    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer=he_init,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer=he_init,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer=he_init,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        256,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer=he_init,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if use_pose:
        input_pose = tf.keras.Input(shape=(3,), name="head_pose")
        x = tf.keras.layers.Concatenate()([x, input_pose])
        inputs = [input_eye, input_pose]
    else:
        inputs = input_eye

    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=he_init)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(2, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005), loss="mse", metrics=["mae"]
    )
    return model


# Training
os.makedirs("models", exist_ok=True)

if USE_CROSS_VALIDATION:
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\n📂 Fold {fold}/{FOLDS}")
        if use_pose:
            inputs_train = {"eye_input": X[train_idx], "head_pose": X_pose[train_idx]}
            inputs_val = {"eye_input": X[val_idx], "head_pose": X_pose[val_idx]}
        else:
            inputs_train = X[train_idx]
            inputs_val = X[val_idx]

        y_train, y_val = y[train_idx], y[val_idx]

        model = build_eye_model()

        callbacks = [
            EarlyStopping(
                monitor="val_mae", patience=7, restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                f"models/fold{fold}_eye_tracking_model_left_relative.keras",
                monitor="val_mae",
                save_best_only=True,
                verbose=1,
            ),
            TensorBoard(log_dir=f"logs/fold{fold}_left"),
        ]

        model.fit(
            inputs_train,
            y_train,
            validation_data=(inputs_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        preds = model.predict(inputs_val)
        plt.figure(figsize=(8, 8))
        plt.scatter(y_val[:, 0], preds[:, 0], alpha=0.5, label="x")
        plt.scatter(y_val[:, 1], preds[:, 1], alpha=0.5, label="y")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Prediction vs Ground Truth – Fold {fold}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"logs/fold{fold}_left_scatter.png")

    print("\n✅ Cross-Validation abgeschlossen. Modelle in 'models/' gespeichert.")

else:
    print("\n📂 Einfaches Training ohne Cross-Validation")
    if use_pose:
        X_train, X_val, Xp_train, Xp_val, y_train, y_val = train_test_split(
            X, X_pose, y, test_size=0.2, random_state=42
        )
        inputs_train = {"eye_input": X_train, "head_pose": Xp_train}
        inputs_val = {"eye_input": X_val, "head_pose": Xp_val}
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        inputs_train = X_train
        inputs_val = X_val

    model = build_eye_model()

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
        inputs_train,
        y_train,
        validation_data=(inputs_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    preds = model.predict(inputs_val)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val[:, 0], preds[:, 0], alpha=0.5, label="x")
    plt.scatter(y_val[:, 1], preds[:, 1], alpha=0.5, label="y")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Ground Truth – Einzeltraining")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/single_run_left_scatter.png")

    print("\n✅ Training abgeschlossen. Modell gespeichert in 'models/'")
