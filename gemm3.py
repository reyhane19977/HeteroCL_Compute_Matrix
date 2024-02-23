import heterocl as hcl
import numpy as np

def extended_gemm(P, Q, R, S, T, alpha, beta, gamma, dtype=hcl.Float()):
    hcl.init(dtype)
    # Placeholders
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((R, S), "C")
    D = hcl.placeholder((S, T), "D")
    E = hcl.placeholder((P, T), "E")

    def kernel(A, B, C, D, E):
        r = hcl.reduce_axis(0, Q, "r")
        s = hcl.reduce_axis(0, S, "s")
        # Matrix multiplication
        AB = hcl.compute((P, R), lambda x, y: hcl.sum(alpha * A[x, r] * B[r, y], axis=r), name="AB")
        CD = hcl.compute((P, T), lambda x, y: hcl.sum(beta * C[y, s] * D[s, x], axis=s), name="CD")
        # Element-wise addition of AB and CD with broadcasting, plus E
        return hcl.compute((P, T), lambda x, y: AB[x, y % R] + CD[x, y] + gamma * E[x, y], name="F")

    # Schedules
    s = hcl.create_schedule([A, B, C, D, E], kernel)
    return hcl.build(s)

# Test function
def test_gemm():
    P, Q, R, S, T = 10, 10, 10, 10, 10
    alpha, beta, gamma = 1.0, 2.0, 3.0
    np_A = np.random.rand(P, Q).astype(np.float32)
    np_B = np.random.rand(Q, R).astype(np.float32)
    np_C = np.random.rand(R, S).astype(np.float32)
    np_D = np.random.rand(S, T).astype(np.float32)
    np_E = np.random.rand(P, T).astype(np.float32)
    np_F = np.zeros((P, T)).astype(np.float32)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)
    hcl_D = hcl.asarray(np_D)
    hcl_E = hcl.asarray(np_E)
    hcl_F = hcl.asarray(np_F)

    f = extended_gemm(P, Q, R, S, T, alpha, beta, gamma)
    f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)

    # Check results
    np_F = hcl_F.asnumpy()
    golden = alpha * np.dot(np_A, np_B) + beta * np.dot(np_C, np_D) + gamma * np_E
    if np.allclose(golden, np_F, rtol=1e-05, atol=1e-06):
        print("Test passed!")
    else:
        print("Test failed.")

# Run the test
test_gemm()