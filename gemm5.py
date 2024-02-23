import heterocl as hcl
import numpy as np

# Define matrix dimensions
P, Q, R, S, T = 10, 20, 30, 40, 50

# Create HeteroCL compute kernel
def compute_kernel(A, B, C, D, E, alpha, beta, gamma):
    F = hcl.compute((P, T), lambda p, t: alpha * hcl.sum(A[p, Q] * B[Q, R] * C[R, S] * D[S, t], axis=[Q, R, S]) + gamma * E[p, t])
    return F

# Create test data (random matrices)
A_np = np.random.rand(P, Q)
B_np = np.random.rand(Q, R)
C_np = np.random.rand(R, S)
D_np = np.random.rand(S, T)
E_np = np.random.rand(P, T)

# Compile the HeteroCL kernel
hcl_A = hcl.placeholder((P, Q))
hcl_B = hcl.placeholder((Q, R))
hcl_C = hcl.placeholder((R, S))
hcl_D = hcl.placeholder((S, T))
hcl_E = hcl.placeholder((P, T))
alpha, beta, gamma = 1.0, 1.0, 1.0
F = compute_kernel(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, alpha, beta, gamma)
s = hcl.create_schedule([F])
target = hcl.platform.zc706
target.config(compile="vivado_hls", mode="csyn")
f = hcl.build(s, target)

# Execute the kernel
hcl_A_np = hcl.asarray(A_np)
hcl_B_np = hcl.asarray(B_np)
hcl_C_np = hcl.asarray(C_np)
hcl_D_np = hcl.asarray(D_np)
hcl_E_np = hcl.asarray(E_np)
hcl_F_np = hcl.asarray(np.zeros((P, T)))
f(hcl_A_np, hcl_B_np, hcl_C_np, hcl_D_np, hcl_E_np, hcl_F_np)

# Verify results
F_np = hcl_F_np.asnumpy()
expected_result = alpha * np.matmul(A_np, B_np) * beta * np.matmul(C_np, D_np) + gamma * E_np
assert np.allclose(F_np, expected_result)

print("HeteroCL implementation is correct!")

# Time the computation (optional)
# Use Linux 'time' command to measure execution time

# Save the Python file
with open("heterocl_matrix_computation.py", "w") as f:
    f.write("# HeteroCL implementation\n")