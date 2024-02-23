import heterocl as hcl
import numpy as np
import time

def gemm_task(P, Q, R, S, T, alpha, beta, gamma, dtype=hcl.Float(), target=None):
    A = hcl.placeholder((P, Q), dtype=dtype, name="A")
    B = hcl.placeholder((Q, R), dtype=dtype, name="B")
    C = hcl.placeholder((R, S), dtype=dtype, name="C")
    D = hcl.placeholder((S, T), dtype=dtype, name="D")
    E = hcl.placeholder((P, T), dtype=dtype, name="E")

    def compute_kernel(A, B, C, D, E):
        r = hcl.reduce_axis(0, Q, name="r")
        F = hcl.compute((P, T), lambda x, y: hcl.sum(alpha * A[x, r] * B[r, y], axis=r, dtype=dtype) +
                                          beta * hcl.sum(C[r, y] * D[r, y], axis=r, dtype=dtype) +
                                          gamma * E[x, y], dtype=dtype, name="F")
        return F

    s = hcl.create_schedule([A, B, C, D, E], compute_kernel)
    f = hcl.build(s, target=target)

    return f

# Example usage
if __name__ == "__main__":
    P, Q, R, S, T = 10, 20, 30, 40, 50
    alpha, beta, gamma = 1.0, 2.0, 3.0

    f = gemm_task(P, Q, R, S, T, alpha, beta, gamma, dtype=hcl.Float(), target="llvm")

    # Generate random matrices for testing
    np_A = np.random.rand(P, Q)
    np_B = np.random.rand(Q, R)
    np_C = np.random.rand(R, S)
    np_D = np.random.rand(S, T)
    np_E = np.random.rand(P, T)

    hcl_A = hcl.asarray(np_A, dtype=hcl.Float())
    hcl_B = hcl.asarray(np_B, dtype=hcl.Float())
    hcl_C = hcl.asarray(np_C, dtype=hcl.Float())
    hcl_D = hcl.asarray(np_D, dtype=hcl.Float())
    hcl_E = hcl.asarray(np_E, dtype=hcl.Float())
    hcl_F = hcl.asarray(np.zeros((P, T)), dtype=hcl.Float())

    f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)

    # Verify correctness
    np_F_expected = alpha * np.dot(np_A, np_B) + beta * np.dot(np_C, np_D) + gamma * np_E
    np.testing.assert_allclose(hcl_F.asnumpy(), np_F_expected, rtol=1e-5)

    print("Matrix F computed successfully!")