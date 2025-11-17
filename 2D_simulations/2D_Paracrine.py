#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 21:51:40 2025

@author: danielpearce
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
size = 200           # Grid size
L = 300              # Simulation size
dt = 0.01             # Time step
steps = 10000        # Number of iterations

#Parameters fixed by non-dimensionalization
kappa = 0
Ki = 1
gammaA = 1
Da = 1

#Parameter variable according to number of neighbours in juxtacrine case: KA
Ka = 1

#Parameters fixed by observation
n = 10
m = 4

#Free parameters
VA = 6          #activator growth rate
VI = 8          #inhibitor growth rate
gammaI = 0.3          #inhibitor degradation rate
Di = 10          #inhibitor diffusion coefficient

N_points = 100   #Number of activated points as initial condition


#Initialize seed of points
def init_point_seeds(size, N_points, VA, seed=None):
    rng = np.random.default_rng(seed)
    if N_points > size*size:
        raise ValueError("N cannot exceed total number of cells.")
    # pick N unique flat indices, then map to (row, col)
    flat_idx = rng.choice(size*size, size=N_points, replace=False)
    rr, cc = np.unravel_index(flat_idx, (size, size))
    A = np.zeros((size, size), dtype=float)
    A[rr, cc] = VA
    return A

def hill(A,I):
    num = np.power(A/Ka,n) + kappa*np.ones(np.shape(A))
    den = np.power(A/Ka,n) + np.power(I/Ki,m) + (1+kappa)*np.ones(np.shape(A))
    return np.divide(num,den)


# Laplacian using finite differences
def laplacian(Z):
    return (
        -4 * Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    )*size*size/(L*L)

# Update function for animation
def update(frame):
    global A, I
    La = laplacian(A)
    Li = laplacian(I)
    H = hill(A,I)

    A += (Da * La + VA*H - gammaA*A) * dt
    I += (Di * Li + VI*H - gammaI*I) * dt

    # Update image
    im.set_data(A)
    ax.set_title(f'Time step: {frame}')
    return [im]


# Initialize concentration grids
A = init_point_seeds(size, N_points, VA * 2, seed=None)
I = np.zeros((size, size), dtype=float)

# Plotting setup
fig, ax = plt.subplots()
im = ax.imshow(A, cmap='inferno', interpolation='bilinear')
plt.axis('off')

# Run animation
ani = FuncAnimation(fig, update, frames=steps, interval=0.05)
plt.show()
