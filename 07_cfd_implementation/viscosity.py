#!/usr/bin/env python3
"""
Mixture viscosity via Cantera.

Defines:
    mu(XO2, XN2, XH2, temperature) -> float  [Pa*s]

Uses mechanism file: mymech.yaml
Assumes ideal-gas mixture and returns Cantera's mixture viscosity (mixture-averaged).
"""

from __future__ import annotations
import cantera as ct
import random 
import numpy as np
import h5py
import os

mech_file = "./mymech.yml"
file_name = "data"
iters = 10000

def _normalize_mole_fractions(xo2: float, xn2: float, xh2: float) -> tuple[float, float, float]:
    vals = [float(xo2), float(xn2), float(xh2)]
    if any(v < 0.0 for v in vals):
        raise ValueError("Mole fractions must be non-negative.")
    s = sum(vals)
    if s <= 0.0:
        raise ValueError("At least one mole fraction must be > 0.")
    return vals[0] / s, vals[1] / s, vals[2] / s


def mu(XO2: float, XN2: float, XH2: float, temperature: float) -> float:
    """
    Return mixture viscosity (Pa*s) for an O2/N2/H2 ideal-gas mixture at given temperature.

    Inputs are treated as mole fractions; if they do not sum to 1, they are normalized.
    Pressure is set to 1 atm (viscosity is weakly pressure-dependent for ideal gases).
    """
    T = float(temperature)
    if T <= 0.0:
        raise ValueError("Temperature must be > 0 K.")

    xo2, xn2, xh2 = _normalize_mole_fractions(XO2, XN2, XH2)

    gas = ct.Solution(mech_file)
    gas.TPY = T, ct.one_atm, {"O2": xo2, "N2": xn2, "H2": xh2}

    return float(gas.viscosity)


if __name__ == "__main__":
    
    # # example
    # val = mu(0.1, 0.8, 0.1, 500.0)
    # print(f"mu = {val:.6e} Pa*s")
    
    # init lists
    xo2_list = []
    xn2_list = []
    xh2_list = []
    t_list = []
    val_list = []

    for i in range(iters):
        # set seed
        random.seed(i)
        np.random.seed(i)

        # generate random mass frac
        ii = random.random()
        jj = random.random()
        kk = random.random()
        T = random.randint(200, 3000)

        # sum and normalize to 1
        s = ii + jj + kk
        xo2_n, xn2_n, xh2_n = ii / s, jj / s, kk / s

        # get visc
        val = mu(ii, jj, kk, T)
        print(f"mu({xo2_n:.3f}, {xn2_n:.3f}, {xh2_n:.3f}, {T}) = {val:.6e} Pa*s")

        # collect normalized mass fractions
        xo2_list.append(xo2_n)
        xn2_list.append(xn2_n)
        xh2_list.append(xh2_n)
        t_list.append(T)
        val_list.append(val)

    # save data to HDF5 file
    data = np.array([xo2_list, xn2_list, xh2_list, t_list, val_list]).T
    with h5py.File(f"{file_name}.h5", "w") as f:
        f.create_dataset("data", data=data)



