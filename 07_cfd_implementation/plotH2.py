#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from viscosity import mu

# load
model = keras.saving.load_model("model/model.keras")
input_mean = np.load("model/input_mean.npy")
input_std  = np.load("model/input_std.npy")
output_mean = np.load("model/output_mean.npy")
output_std  = np.load("model/output_std.npy")

# constant mass fraction
xo2 = 0.23
xn2 = 0.77
T = 1000

# init T step
xh2 = np.arange(0, 1, 0.01) 
y_true = []
y_pred = []

for h in xh2:
    
    # true data
    new_xo2 = xo2 * (1 - h)
    new_xn2 = xn2 * (1 - h)
    visc_true = mu(new_xo2, new_xn2, h, T)
    y_true.append(visc_true)
    
    # mlp data
    input_data = np.array([[new_xo2, new_xn2, h, T]], dtype=np.float64)
    input_scaled = (input_data - input_mean) / input_std
    pred_scaled = model.predict(input_scaled, verbose=0)
    yy = pred_scaled * output_std + output_mean
    y_pred.append(yy[0][0])

# plot data
# https://matplotlib.org/stable/api/markers_api.html
plt.figure(figsize=(8, 6))
# plt.plot(temperatures, y_true, label='true viscosity (cantera)', color='blue', marker='.')
# plt.plot(temperatures, y_pred, label='mlp viscosity', color='orange', marker='.')
plt.semilogy(xh2, y_true, label='true viscosity (cantera)', color='blue', marker='.')
plt.semilogy(xh2, y_pred, label='mlp viscosity', color='orange', marker='.')
plt.xlabel('H2 mass fraction')
plt.ylabel('Viscosity (Pa·s)')
plt.title(f'mlp vs true (xo2={xo2}, xn2={xn2}, T={T}K)')
plt.legend()
plt.grid(True)
plt.show()