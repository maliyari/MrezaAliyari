import numpy as np

# Fix the random seed (last digit of your ID)
np.random.seed(7)

# 1. Construct a random 5x5 matrix P and normalize each row
P = np.random.rand(5, 5)
P /= P.sum(axis=1, keepdims=True)

# 2. Construct a random size-5 vector p and normalize it
p = np.random.rand(5)
p /= p.sum()

# Apply the transition rule 50 times to obtain p_50
for i in range(50):
    p = np.dot(P.T, p)

# 3. Compute the eigenvector v of P^T corresponding to the eigenvalue 1
eigenvalues, eigenvectors = np.linalg.eig(P.T)
v = eigenvectors[:, np.isclose(eigenvalues, 1)]

# Scale the eigenvector v so that its components sum to 1
v /= v.sum()

# 4. Compute the component-wise difference between p_50 and the stationary distribution v
diff = np.abs(p - v)

# Check if the difference is less than 1e-5 for all components
if np.all(diff < 1e-5):
    print("The component-wise difference between p_50 and the stationary distribution is less than 1e-5.")
else:
    print("The component-wise difference between p_50 and the stationary distribution is greater than or equal to 1e-5.")
