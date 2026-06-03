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
xo2 = 0.3
xn2 = 0.2
xh2 = 0.5

# init T step
temperatures = np.arange(200, 3000, 10) 
y_true = []
y_pred = []

for T in temperatures:
    
    # true data
    visc_true = mu(xo2, xn2, xh2, T)
    y_true.append(visc_true)
    
    # mlp data
    input_data = np.array([[xo2, xn2, xh2, T]], dtype=np.float64)
    input_scaled = (input_data - input_mean) / input_std
    pred_scaled = model.predict(input_scaled, verbose=0)
    yy = pred_scaled * output_std + output_mean
    y_pred.append(yy[0][0])

# plot data
# https://matplotlib.org/stable/api/markers_api.html
plt.figure(figsize=(8, 6))
# plt.plot(temperatures, y_true, label='true viscosity (cantera)', color='blue', marker='.')
# plt.plot(temperatures, y_pred, label='mlp viscosity', color='orange', marker='.')
plt.semilogy(temperatures, y_true, label='true viscosity (cantera)', color='blue', marker='.')
plt.semilogy(temperatures, y_pred, label='mlp viscosity', color='orange', marker='.')
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (Pa·s)')
plt.title(f'mlp vs true (xo2={xo2}, xn2={xn2}, xh2={xh2})')
plt.legend()
plt.grid(True)
plt.show()