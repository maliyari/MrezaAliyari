from sympy import *
import numpy as np
import matplotlib.pyplot as plt

# Define the symbols for differentiation
x, y = symbols('x y')

# Define the function to be differentiated
f = sin(x)**2 * exp(x) * cos(y)

# Task 1: Find the closed form expression of f_xy(x, y)
f_xy = diff(f, x, y)  # Calculate the second mixed partial derivative
print("f_xy(x, y) = ", f_xy)

# Task 2: Find f_xy(2, 3) with sympy and take 15 significant digits
f_xy_exact = f_xy.subs([(x, 2), (y, 3)])  # Substitute x=2 and y=3 into the expression
f_xy_exact = N(f_xy_exact, 15)  # Convert to a numerical value with 15 significant digits
print("f_xy(2, 3) = ", f_xy_exact)

# Task 3: Approximate f_xy(2, 3) using central difference approximation
x0 = 2
y0 = 3
h = [0.1, 0.01, 0.001, 0.0001]  # List of different step sizes
f_xy_approx = []  # To store the numerical approximations

# Loop through different step sizes and calculate the approximations
for i in range(len(h)):
    f_xplus_h_yplus_h = f.subs([(x, x0+h[i]), (y, y0+h[i])])  # f(x0 + h, y0 + h)
    f_xplus_h_yminus_h = f.subs([(x, x0+h[i]), (y, y0-h[i])])  # f(x0 + h, y0 - h)
    f_xminus_h_yplus_h = f.subs([(x, x0-h[i]), (y, y0+h[i])])  # f(x0 - h, y0 + h)
    f_xminus_h_yminus_h = f.subs([(x, x0-h[i]), (y, y0-h[i])])  # f(x0 - h, y0 - h)
    f_xy_approx.append((f_xplus_h_yplus_h - f_xplus_h_yminus_h - f_xminus_h_yplus_h + f_xminus_h_yminus_h) / (4 * h[i]))
    print("Numerical Differentiation Result (h = {}): {}".format(h[i], f_xy_approx[i]))

# Task 4: Plot the absolute error
error = np.abs(f_xy_exact - np.array(f_xy_approx))
plt.plot(h, error)
plt.gca().invert_xaxis()  # Invert x-axis to show decreasing step size
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.xlabel('Step Size h')  # Label for the x-axis
plt.ylabel('Absolute Error')  # Label for the y-axis
plt.title('Absolute Error vs. Step Size h for Central Difference Approximation')  # Title for the plot
plt.show()  # Display the plot
