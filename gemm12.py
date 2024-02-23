import heterocl as hcl
import numpy as np

def compute_kernel(A, B, C, D, E, alpha, beta, gamma):
    P, Q = A.shape
    R = B.shape[1]
    S = C.shape[1]
    T = D.shape[1]

    F = hcl.compute((P, T), lambda x, y: alpha * hcl.sum(A[x, r] * B[r, y], axis=r) * beta * hcl.sum(C[r, y] * D[r, z], axis=r) + gamma * E[x, z], dtype=hcl.Float())
    return F

def test_bench():
    P, Q, R, S, T = 10, 20, 30, 40, 50
    alpha, beta, gamma = 1.0, 2.0, 3.0

    A = np.random.rand(P, Q)
    B = np.random.rand(Q, R)
    C = np.random.rand(R, S)
    D = np.random.rand(S, T)
    E = np.random.rand(P, T)

    hcl_A = hcl.asarray(A, dtype=hcl.Float())
    hcl_B = hcl.asarray(B, dtype=hcl.Float())
    hcl_C = hcl.asarray(C, dtype=hcl.Float())
    hcl_D = hcl.asarray(D, dtype=hcl.Float())
    hcl_E = hcl.asarray(E, dtype=hcl.Float())

    F = compute_kernel(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, alpha, beta, gamma)

    # Execute the computation
    s = hcl.create_schedule([hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, F])
    f = hcl.build(s)
    f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, F)

    # Convert result back to NumPy array
    result = F.asnumpy()
    print("Result matrix F:\n", result)