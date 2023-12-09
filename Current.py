# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 02:47:54 2023

@author: ASUS
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Solving the Laplace Equation for Obtaining Streamline Field in a 2D Forward Step
# Specifying parameters
x1 = 95.0
x2 = 57.0
h1 = 45
h2 = 27
time = 200


q = 45.0
u_inlet = q / (h1 * 1.0)
u_outlet = q / (h2 * 1.0)

nx = 153
ny = 46

del_y = h1 / (ny - 1)
del_x = (x1 + x2) / (nx - 1)
beta = (del_x / del_y) ** 2

# Grid
y = np.linspace(0, h1, ny).reshape(-1, 1)
y = np.tile(y, (1, nx))

x = np.linspace(0, x1 + x2, nx).reshape(1, -1)
x = np.tile(x, (ny, 1))

# Applying initial value to variables
error = 1.0

initial_s = np.zeros((ny, nx))
final_s = np.zeros((ny, nx))
ekhtelaf_s = np.zeros((ny, nx))
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

# Inlet Boundary Condition
for i in range(1, ny):
    initial_s[i, 0] = initial_s[i - 1, 0] + (u_inlet * del_y)
    final_s[i, 0] = final_s[i - 1, 0] + (u_inlet * del_y)

initial_s[-1, :] = initial_s[-1, 0]
final_s[-1, :] = final_s[-1, 0]

# Outlet Boundary Condition
for i in range(ny - 1, int(ny - (h2 / del_y)) - 1, -1):
    initial_s[i - 1, -1] = initial_s[i, -1] - (u_outlet * del_y)
    final_s[i - 1, -1] = final_s[i, -1] - (u_outlet * del_y)

# Iterative solving the streamline field
while error > 1e-10:
    initial_s = final_s.copy()

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if x[i, j] < x1:
                final_s[i, j] = (
                    initial_s[i, j + 1]
                    + initial_s[i, j - 1]
                    + beta * (initial_s[i - 1, j] + initial_s[i + 1, j])
                ) / (2 * (1 + beta))
            elif x[i, j] >= x1 and y[i, j] > (h1 - h2):
                final_s[i, j] = (
                    initial_s[i, j + 1]
                    + initial_s[i, j - 1]
                    + beta * (initial_s[i - 1, j] + initial_s[i + 1, j])
                ) / (2 * (1 + beta))

    ekhtelaf_s = final_s - initial_s
    error = np.max(np.abs(ekhtelaf_s))

# Calculating the velocity field
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
u[1:, 0] = u_inlet

for i in range(1, ny):
    for j in range(1, nx):
        u[i, j] = (final_s[i, j] - final_s[i - 1, j]) / del_y
        v[i, j] = -(final_s[i, j] - final_s[i, j - 1]) / del_x

# Exporting output for Tecplot
a = np.zeros(((nx) * (ny), 4))

k = 0
for i in range(ny):
    for j in range(nx):
        a[k, 0] = x[i, j]
        a[k, 1] = y[i, j]
        a[k, 2] = u[i, j]
        a[k, 3] = v[i, j]
        k += 1


# Handle non-finite values in x, y, and u
x = np.ma.masked_invalid(x).filled(np.nan)
y = np.ma.masked_invalid(y).filled(np.nan)
u = np.ma.masked_invalid(u).filled(np.nan)

# Create a pseudocolor plot
pcm = plt.pcolormesh(x, y, u, cmap='jet')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Velocity Field')

# Add colorbar for reference
plt.colorbar(pcm)

# Adjust the aspect ratio of the axes for a better representation
plt.axis('equal')
plt.axis('tight')

# Display the grid (optional)
plt.grid(False)