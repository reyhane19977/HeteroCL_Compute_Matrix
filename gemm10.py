import heterocl as hcl
import numpy as np
import time

def compute_kernel(A, B, C, D, E, alpha, beta, gamma):
    P, Q, R, S, T = A.shape[0], A.shape[1], B.shape[1], C.shape[1], E.shape[1]

    # Compute intermediate results
    AB = hcl.compute((P, R), lambda x, y: hcl.sum(alpha * A[x, r] * B[r, y], axis=r), name="AB")
    CD = hcl.compute((R, T), lambda x, y: hcl.sum(beta * C[x, s] * D[s, y], axis=s), name="CD")

    # Compute final result F
    F = hcl.compute((P, T), lambda x, y: AB[x, r] * CD[r, s] + gamma * E[x, y], name="F")

    return F

def main():
    P, Q, R, S, T = 16, 20, 18, 22, 24
    alpha, beta, gamma = 0.1, 0.2, 0.3

    A = np.random.rand(P, Q).astype(np.float32)
    B = np.random.rand(Q, R).astype(np.float32)
    C = np.random.rand(R, S).astype(np.float32)
    D = np.random.rand(S, T).astype(np.float32)
    E = np.random.rand(P, T).astype(np.float32)

    # Initialize HeteroCL environment
    hcl.init()

    # Create placeholders for input matrices
    A_hcl = hcl.placeholder((P, Q), "A")
    B_hcl = hcl.placeholder((Q, R), "B")
    C_hcl = hcl.placeholder((R, S), "C")
    D_hcl = hcl.placeholder((S, T), "D")
    E_hcl = hcl.placeholder((P, T), "E")

    # Compute kernel
    F_hcl = compute_kernel(A_hcl, B_hcl, C_hcl, D_hcl, E_hcl, alpha, beta, gamma)

    # Build and run the computation
    f = hcl.build(hcl.create_schedule([A_hcl, B_hcl, C_hcl, D_hcl, E_hcl], F_hcl))
    hcl_result = hcl.asarray(np.zeros((P, T), dtype=np.float32))
    f(A, B, C, D, E, hcl_result)

    # Verify correctness
    np_result = alpha * np.matmul(A, B) + beta * np.matmul(C, D) + gamma * E
    np.testing.assert_allclose(hcl_result.asnumpy(), np_result, rtol=1e-5)
    print("Results match!")

if __name__ == "__main__":
    main()