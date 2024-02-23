import heterocl as hcl
import numpy as np

def compute_kernel(alpha, beta, gamma, A, B, C, D, E):
    P, Q = A.shape
    R, S = C.shape
    T, _ = D.shape

    hcl.init()
    hcl_A = hcl.placeholder((P, Q), "A")
    hcl_B = hcl.placeholder((Q, R), "B")
    hcl_C = hcl.placeholder((R, S), "C")
    hcl_D = hcl.placeholder((S, T), "D")
    hcl_E = hcl.placeholder((P, T), "E")

    def kernel(A, B, C, D, E):
        F = hcl.compute((P, T), lambda i, j: 0, "F")
        with hcl.for_(0, P, name="i") as i:
            with hcl.for_(0, T, name="j") as j:
                with hcl.for_(0, Q, name="k") as k:
                    with hcl.for_(0, S, name="l") as l:
                        F[i, j] += alpha * A[i, k] * B[k, j] * beta * C[k, l] * D[l, j] + gamma * E[i, j]
        return F

    s = hcl.create_schedule([hcl_A, hcl_B, hcl_C, hcl_D, hcl_E], kernel)
    f = hcl.build(s)

    hcl_F = hcl.asarray(np.zeros((P, T)), dtype=hcl.Float())
    f(hcl.asarray(A), hcl.asarray(B), hcl.asarray(C), hcl.asarray(D), hcl.asarray(E), hcl_F)

    return hcl_F.asnumpy()

# Testbench
if __name__ == "__main__":
    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    P, Q, R, S, T = 4, 3, 5, 2, 4
    A = np.random.rand(P, Q)
    B = np.random.rand(Q, R)
    C = np.random.rand(R, S)
    D = np.random.rand(S, T)
    E = np.random.rand(P, T)

    F = compute_kernel(alpha, beta, gamma, A, B, C, D, E)
    print("Resulting matrix F:")
    print(F)