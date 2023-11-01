import numpy as np

# Part A

# Define a function called 'LUdecomposition' that takes a matrix 'A' as input
def LUdecomposition(A):
    n = A.shape[0]  # Get the number of rows (assuming A is a square matrix)
    L = np.eye(n)  # Initialize an identity matrix of the same size as A as 'L'
    U = A.copy()  # Create a copy of matrix 'A' as 'U'

    # Loop through the rows of 'U' for LU decomposition
    for i in range(n - 1):
        if U[i, i] == 0:
            raise ValueError("Zero pivot encountered. Pivoting is required.")  # Check for zero pivot
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]  # Calculate the multiplier for the lower triangular matrix
            U[j, i:] -= L[j, i] * U[i, i:]  # Perform row operations to eliminate elements in 'U'

    # Return the lower triangular matrix 'L' and the upper triangular matrix 'U'
    return L, U

# Create several random matrices for testing
A1 = np.random.rand(3, 3)
A2 = np.random.rand(4, 4)
A3 = np.random.rand(5, 5)

# Obtain the L and U matrices for each matrix and perform a matrix multiplication to check the decomposition
L1, U1 = LUdecomposition(A1)
print("A1: \n", A1)
print("L1: \n", L1)
print("U1: \n", U1)
print("A1 = L1U1: \n", np.allclose(A1, np.dot(L1, U1)))  # Check if A1 equals the product of L1 and U1

L2, U2 = LUdecomposition(A2)
print("A2: \n", A2)
print("L2: \n", L2)
print("U2: \n", U2)
print("A2 = L2U2: \n", np.allclose(A2, np.dot(L2, U2)))  # Check if A2 equals the product of L2 and U2

L3, U3 = LUdecomposition(A3)
print("A3: \n", A3)
print("L3: \n", L3)
print("U3: \n", U3)
print("A3 = L3U3: \n", np.allclose(A3, np.dot(L3, U3)))  # Check if A3 equals the product of L3 and U3

# Part B

# Define a function called 'pivoting' that takes a matrix 'A' as input
def pivoting(A):
    n = A.shape[0]  # Get the number of rows (assuming A is a square matrix)
    P = np.eye(n)  # Initialize an identity matrix of the same size as A as 'P'

    # Loop through the rows of 'A' for row pivoting
    for i in range(n - 1):
        # Find the row index with the maximum absolute value in column 'i'
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            # Swap the rows of the permutation matrix 'P' to perform row pivoting
            P[[i, max_row]] = P[[max_row, i]]

    return P

# Create an example matrix 'A' that would raise an error in LU decomposition
A = np.array([[0, 1], [1, 0]])

try:
    L, U = LUdecomposition(A)  # Attempt LU decomposition (will raise an error)
except ValueError as e:
    print(e)  # Print the error message

# Use pivoting() to find permutation matrix 'P', and find the pure LU decomposition of PA
P = pivoting(A)  # Find the permutation matrix 'P' to handle zero pivot
PA = np.dot(P, A)  # Apply permutation 'P' to 'A'
L, U = LUdecomposition(PA)  # Perform LU decomposition on PA
print("A: \n", A)
print("P: \n", P)
print("PA: \n", PA)
print("L: \n", L)
print("U: \n", U)
print("PA = LU: \n", np.allclose(PA, np.dot(L, U)))  # Check if PA equals LU
