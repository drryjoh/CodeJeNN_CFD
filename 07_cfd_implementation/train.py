#!/usr/bin/env python3
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import sys
import time
import random
import warnings
import math
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import joblib
import h5py

def main():
    #mlp stuff
    INPUT_DIM = 4
    OUTPUT_DIM = 1
    SPLIT = 0.2
    LR = 1e-3
    EPOCHS = 500
    BS = 32
    NEURONS = 12
    BIT = np.float64
    FILE = "mu_training_data.csv"
    MODEL_DIR = "model"

    #seed stuff
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # #load data from HDF5
    # with h5py.File(FILE, "r") as f:
    #     data = np.array(f["data"])
    # X = data[:, :4].astype(BIT)
    # y = data[:, 4:5].astype(BIT) 

    #load data from csv file
    df = pd.read_csv(FILE)
    X = df[['O2', 'N2', 'H2', 'T']].values.astype(BIT)
    y = df[['mu']].values.astype(BIT)

    #get number of sampels
    NUM_SAMPLES = X.shape[0]
    print(f'found {NUM_SAMPLES} samples.')

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=SPLIT, random_state=SEED
    )

    #scale inputs and outputs
    #https://scikit-learn.org/stable/modules/preprocessing.html
    x_norm = sklearn.preprocessing.StandardScaler()
    X_train = x_norm.fit_transform(X_train)
    X_val = x_norm.transform(X_val)

    y_norm = sklearn.preprocessing.StandardScaler()
    y_train = y_norm.fit_transform(y_train)
    y_val = y_norm.transform(y_val)

    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_DIM,), dtype=BIT),

        keras.layers.Dense(NEURONS, activation="tanh"),
        keras.layers.Dense(NEURONS, activation="tanh"),
        keras.layers.Dense(NEURONS, activation="tanh"),
        keras.layers.Dense(OUTPUT_DIM, activation=None)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=25)
    ]

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    loss, mse = model.evaluate(X_val, y_val)

    # save model and scalers
    # https://stackoverflow.com/questions/54879434/how-to-use-pickle-to-save-sklearn-model
    os.mkdir(MODEL_DIR)
    model.save(f"{MODEL_DIR}/model_hi_NEW.keras")
    np.save(f"{MODEL_DIR}/input_mean.npy", x_norm.mean_)
    np.save(f"{MODEL_DIR}/input_std.npy", x_norm.scale_)
    np.save(f"{MODEL_DIR}/output_mean.npy", y_norm.mean_)
    np.save(f"{MODEL_DIR}/output_std.npy", y_norm.scale_)

if __name__ == "__main__":
    main()
