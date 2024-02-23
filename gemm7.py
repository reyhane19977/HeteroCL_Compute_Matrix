import heterocl as hcl
import numpy as np

def compute_matrix(alpha, beta, gamma, A, B, C, D, E):
    # Define HeteroCL schedule
    hcl.init()
    P, Q, R, S, T = A.shape[0], A.shape[1], B.shape[1], C.shape[1], E.shape[1]

    # Create placeholders for input matrices
    A_mat = hcl.placeholder((P, Q), name="A")
    B_mat = hcl.placeholder((Q, R), name="B")
    C_mat = hcl.placeholder((R, S), name="C")
    D_mat = hcl.placeholder((S, T), name="D")
    E_mat = hcl.placeholder((P, T), name="E")

    # Create output matrix F
    F_mat = hcl.compute((P, T), lambda p, t: alpha * hcl.reduce_axis(Q)(A_mat[p, q] * B_mat[q, r], axis_name="q") *
                                          beta * hcl.reduce_axis(S)(C_mat[r, s] * D_mat[s, t], axis_name="s") +
                                          gamma * E_mat[p, t], name="F")

    # Create schedule
    s = hcl.create_schedule([A_mat, B_mat, C_mat, D_mat, E_mat, F_mat])

    # Optimize: Parallelize and unroll loops
    s[F_mat].parallel(F_mat.axis[0])
    s[F_mat].unroll(F_mat.axis[1])

    # Build the kernel
    target = hcl.platform.zc706
    s = hcl.build(s, target=target)

    # Create input data
    A_data = np.random.rand(P, Q)
    B_data = np.random.rand(Q, R)
    C_data = np.random.rand(R, S)
    D_data = np.random.rand(S, T)
    E_data = np.random.rand(P, T)

    # Execute the kernel
    hcl_A = hcl.asarray(A_data)
    hcl_B = hcl.asarray(B_data)
    hcl_C = hcl.asarray(C_data)
    hcl_D = hcl.asarray(D_data)
    hcl_E = hcl.asarray(E_data)
    hcl_F = hcl.asarray(np.zeros((P, T)))

    s(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)

    # Retrieve the result
    result = hcl_F.asnumpy()
    print("Result matrix F:")
    print(result)

# Example dimensions
P, Q, R, S, T = 3, 4, 5, 6, 7

# Constants
alpha = 1.0
beta = 2.0
gamma = 0.5

# Generate random matrices 
A = np.random.rand(P, Q)
B = np.random.rand(Q, R)
C = np.random.rand(R, S)
D = np.random.rand(S, T)
E = np.random.rand(P, T)

# Compute F
compute_matrix(alpha, beta, gamma, A, B, C, D, E)