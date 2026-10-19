import os
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_PATH = os.path.join(BASE_DIR, "dataset.zip")
DATA_DIR = os.path.join(BASE_DIR, "UCI HAR Dataset")

CLASS_NAMES = ["Walking", "Walking Up", "Walking Down", "Sitting", "Standing", "Laying"]
N_CLASSES = len(CLASS_NAMES)

np.random.seed(42)
tf.random.set_seed(42)


def download_and_extract():
    if not os.path.isdir(DATA_DIR):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(BASE_DIR)
    print("Dataset ready")


def load_data():
    X_train_full = pd.read_csv(os.path.join(DATA_DIR, "train", "X_train.txt"), header=None, sep=r"\s+")
    y_train_full = pd.read_csv(os.path.join(DATA_DIR, "train", "y_train.txt"), header=None, sep=r"\s+")
    X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "test", "X_test.txt"), header=None, sep=r"\s+")
    y_test = pd.read_csv(os.path.join(DATA_DIR, "test", "y_test.txt"), header=None, sep=r"\s+")
    subjects = pd.read_csv(os.path.join(DATA_DIR, "train", "subject_train.txt"), header=None, sep=r"\s+").values.ravel()

    y_train_full = np.asarray(y_train_full).ravel() - 1
    y_test = np.asarray(y_test).ravel() - 1
    return X_train_full, y_train_full, X_test_raw, y_test, subjects


def main():
    download_and_extract()
    X_train_full, y_train_full, X_test_raw, y_test, subjects = load_data()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X_train_full, y_train_full, groups=subjects))

    X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test_raw)

    model = keras.Sequential([
        keras.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(N_CLASSES, activation="softmax"),
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64,
                        callbacks=[early_stop], verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(y_test, y_pred))

    model.save(os.path.join(BASE_DIR, "har_model.keras"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "har_scaler.joblib"))
    X_test_raw.to_csv(os.path.join(BASE_DIR, "X_test_raw.csv"), index=False, header=False)
    pd.DataFrame(y_test).to_csv(os.path.join(BASE_DIR, "y_test.csv"), index=False, header=False)
    print("Saved har_model.keras, har_scaler.joblib, X_test_raw.csv, y_test.csv")


if __name__ == "__main__":
    main()
