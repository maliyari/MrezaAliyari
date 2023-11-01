import numpy as np
from sympy import *
import matplotlib.pyplot as plt

# Define the symbol for integration
x = symbols('x')

# Define the function to be integrated symbolically
f = 3*x**2 + 2*x + 2

# Part 1: Symbolic Integration
# Compute the symbolic integral of the function
F = integrate(f, x)
print("Symbolic Integration Result: ", F)

# Part 2: Numerical Integration using Trapezoid Rule
# Define the integration limits and a list of different partition sizes (N)
a = -4
b = 6
N = [10, 20, 40, 80, 160, 320, 640, 1280]
I = []  # To store the numerical approximations

# Loop through different values of N and calculate the approximations
for n in N:
    h = (b - a) / n  # Calculate the step size
    x_vals = np.linspace(a, b, n+1)  # Generate equally spaced x values
    y_vals = [f.subs(x, x_val) for x_val in x_vals]  # Evaluate the function at x values
    y_vals[0] /= 2  # Adjust the first and last values for the trapezoid rule
    y_vals[-1] /= 2
    I.append(h * np.sum(y_vals))  # Calculate the approximation using the trapezoid rule
    print("N =", n, "Approximation =", I[-1])

# Part 3: Plot the absolute error
# Calculate the exact integral value
I_exact = F.subs(x, b) - F.subs(x, a)

# Calculate the absolute error for each approximation
error = np.abs(I_exact - np.array(I))

# Create a plot to visualize the absolute error
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(N, error)
ax.set_xscale('log')  # Set the x-axis to a logarithmic scale
ax.set_yscale('log')  # Set the y-axis to a logarithmic scale
ax.set_xlabel('N')  # Label for the x-axis
ax.set_ylabel('Absolute Error')  # Label for the y-axis
ax.set_title('Absolute Error vs. N for Trapezoid Rule')  # Title for the plot
plt.show()  # Display the plot
