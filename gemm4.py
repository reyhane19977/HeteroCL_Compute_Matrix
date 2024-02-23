import heterocl as hcl
import numpy as np

def matrix_multiply(alpha, beta, gamma, A, B, C, D, E):
    P, Q = A.shape
    R = B.shape[1]
    S = C.shape[1]
    T = D.shape[1]

    hcl.init()
    hcl_A = hcl.placeholder((P, Q))
    hcl_B = hcl.placeholder((Q, R))
    hcl_C = hcl.placeholder((R, S))
    hcl_D = hcl.placeholder((S, T))
    hcl_E = hcl.placeholder((P, T))

    def kernel(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E):
        AB = hcl.compute((P, R), lambda i, j: hcl.sum(hcl_A[i, k] * hcl_B[k, j] for k in range(Q)), name="AB")
        CD = hcl.compute((R, T), lambda i, j: hcl.sum(hcl_C[i, k] * hcl_D[k, j] for k in range(S)), name="CD")
        F = hcl.compute((P, T), lambda i, j: alpha * AB[i, j] * beta * CD[i, j] + gamma * hcl_E[i, j], name="F")
        return F

    s = hcl.create_schedule([hcl_A, hcl_B, hcl_C, hcl_D, hcl_E], kernel)
    return hcl.build(s), (hcl_A, hcl_B, hcl_C, hcl_D, hcl_E)


def test():
    alpha = 1.5
    beta = 2.5
    gamma = 3.5
    P, Q, R, S, T = 4, 5, 6, 7, 8
    A = np.random.rand(P, Q)
    B = np.random.rand(Q, R)
    C = np.random.rand(R, S)
    D = np.random.rand(S, T)
    E = np.random.rand(P, T)

    f, placeholders = matrix_multiply(alpha, beta, gamma, A, B, C, D, E)
    hcl_A, hcl_B, hcl_C, hcl_D, hcl_E = placeholders

    hcl_F = hcl.asarray(np.zeros((P, T)))
    f(hcl.asarray(A), hcl.asarray(B), hcl.asarray(C), hcl.asarray(D), hcl.asarray(E), hcl_F)

    F = hcl_F.asnumpy()
    expected_F = alpha * np.dot(A, B) * beta * np.dot(C, D) + gamma * E
    assert np.allclose(F, expected_F), "Test failed!"

    print("Test passed!")

test()


def matrix_multiply_optimized(alpha, beta, gamma, A, B, C, D, E):
    P, Q = A.shape
    R = B.shape[1]
    S = C.shape[1]
    T = D.shape[1]

    hcl.init()
    hcl_A = hcl.placeholder((P, Q))
    hcl_B = hcl.placeholder((Q, R))
    hcl_C = hcl.placeholder((R, S))
    hcl_D = hcl.placeholder((S, T))
    hcl_E = hcl.placeholder((P, T))

    def kernel(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E):
        AB = hcl.compute((P, R), lambda i, j: hcl.sum(hcl_A[i, k] * hcl_B[k, j] for k in range(Q)), name="AB")
        CD = hcl.compute((R, T), lambda i, j: hcl.sum(hcl_C[i, k] * hcl_D[k, j] for k in range(S)), name="CD")
        F = hcl.compute((P, T), lambda i, j: alpha * AB[i, j] * beta * CD[i, j] + gamma * hcl_E[i, j], name="F")
        return F

    s = hcl.create_schedule([hcl_A, hcl_B, hcl_C, hcl_D, hcl_E], kernel)
    s[AB].unroll(AB.axis[1])
    s[CD].unroll(CD.axis[1])
    return hcl.build(s)