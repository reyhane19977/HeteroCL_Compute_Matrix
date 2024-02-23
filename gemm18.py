import heterocl as hcl
import numpy as np

def matrix_multiply(alpha, beta, gamma, A_shape, B_shape, C_shape, D_shape, E_shape):
    P, Q = A_shape
    _, R = B_shape
    _, S = C_shape
    _, T = D_shape
    hcl.init()

    hcl_A = hcl.placeholder(A_shape)
    hcl_B = hcl.placeholder(B_shape)
    hcl_C = hcl.placeholder(C_shape)
    hcl_D = hcl.placeholder(D_shape)
    hcl_E = hcl.placeholder(E_shape)

    def kernel(A, B, C, D, E):
    r_AB = hcl.reduce_axis(0, Q, name='r_AB')
    r_CD = hcl.reduce_axis(0, S, name='r_CD')
    r_F = hcl.reduce_axis(0, R, name='r_F')
    AB = hcl.compute((P, R), lambda i, j: hcl.sum(A[i, r_AB] * B[r_AB, j], axis=r_AB), name="AB")
    CD = hcl.compute((R, T), lambda i, j: hcl.sum(C[i, r_CD] * D[r_CD, j], axis=r_CD), name="CD")
    F = hcl.compute((P, T), lambda i, j: alpha * hcl.sum(AB[i, r_F] * CD[r_F, j], axis=r_F) + gamma * E[i, j], name="F")
    return F



    s = hcl.create_schedule([hcl_A, hcl_B, hcl_C, hcl_D, hcl_E], kernel)
    return hcl.build(s)

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

    f = matrix_multiply(alpha, beta, gamma, A.shape, B.shape, C.shape, D.shape, E.shape)

    hcl_A = hcl.asarray(A)
    hcl_B = hcl.asarray(B)
    hcl_C = hcl.asarray(C)
    hcl_D = hcl.asarray(D)
    hcl_E = hcl.asarray(E)
    hcl_F = hcl.asarray(np.zeros((P, T)))

    f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)

    F_heterocl = hcl_F.asnumpy()
    F_expected = alpha * np.dot(A, np.dot(B, beta * np.dot(C, D))) + gamma * E

    if not np.allclose(F_heterocl, F_expected, atol=1e-5):
        print("Test failed!")
        print("Computed F:")
        print(F_heterocl)
        print("Expected F:")
        print(F_expected)
    else:
        print("Test passed!")

test()