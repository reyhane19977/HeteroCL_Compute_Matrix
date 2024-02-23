import numpy as np

def compute_F(A, B, C, D, E, alpha, beta, gamma):
    # Perform matrix multiplications
    AB = np.matmul(A, B)
    CD = np.matmul(C, D)
    F = alpha * np.matmul(AB, beta * CD) + gamma * E
    return F

# Example dimensions
P, Q, R, S, T = 3, 4, 5, 6, 7

# Generate random matrices
A = np.random.rand(P, Q)
B = np.random.rand(Q, R)
C = np.random.rand(R, S)
D = np.random.rand(S, T)
E = np.random.rand(P, T)

# Constants
alpha = 1.0
beta = 2.0
gamma = 0.5

# Compute F
F_naive = compute_F(A, B, C, D, E, alpha, beta, gamma)

# Print the result
print("Naive F:\n", F_naive)


# Test bench in HeteroCL
from heterocl import *

def test_bench():
    hcl.init()

    # Define input tensors
    A_tensor = hcl.placeholder((P, Q), "A")
    B_tensor = hcl.placeholder((Q, R), "B")
    C_tensor = hcl.placeholder((R, S), "C")
    D_tensor = hcl.placeholder((S, T), "D")
    E_tensor = hcl.placeholder((P, T), "E")

    # Define constants
    alpha_const = hcl.scalar(alpha)
    beta_const = hcl.scalar(beta)
    gamma_const = hcl.scalar(gamma)

    # Compute F
    F_tensor = hcl.compute((P, T), lambda i, j: alpha_const[0] * (A_tensor[i, :] @ B_tensor[:, j]) +
                                              beta_const[0] * (C_tensor[i, :] @ D_tensor[:, j]) +
                                              gamma_const[0] * E_tensor[i, j], "F")

    # Create a test harness
    s = hcl.create_schedule([A_tensor, B_tensor, C_tensor, D_tensor, E_tensor], F_tensor)
    f = hcl.build(s)

    # Run the test
    np_A = np.random.rand(P, Q)
    np_B = np.random.rand(Q, R)
    np_C = np.random.rand(R, S)
    np_D = np.random.rand(S, T)
    np_E = np.random.rand(P, T)
    np_F = np.zeros((P, T))

    f(hcl.asarray(np_A), hcl.asarray(np_B), hcl.asarray(np_C), hcl.asarray(np_D), hcl.asarray(np_E), hcl.asarray(np_F))

    print("HeteroCL F:\n", hcl.asnumpy(np_F))

if __name__ == "__main__":
    test_bench()
