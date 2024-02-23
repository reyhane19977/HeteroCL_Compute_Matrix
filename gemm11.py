import heterocl as hcl
import numpy as np

def matrix_computation(P, Q, R, S, T, alpha, beta, gamma):
    # Initialize HeteroCL environment
    hcl.init()

    # Define input matrices
    A = hcl.placeholder((P, Q), name="A")
    B = hcl.placeholder((Q, R), name="B")
    C = hcl.placeholder((R, S), name="C")
    D = hcl.placeholder((S, T), name="D")
    E = hcl.placeholder((P, T), name="E")

    # Define output matrix F
    F = hcl.compute((P, T), lambda p, t: alpha * hcl.sum(A[p, q] * B[q, r] * C[r, s] * D[s, t],
                                                        axis=[q, r, s]) + gamma * E[p, t],
                    name="F")

    # Create schedule
    s = hcl.create_schedule([A, B, C, D, E, F])

    # Compute optimizations
    s[F].parallel(F.axis[0])
    s[F].unroll(F.axis[1])

    # Compile to FPGA or CPU
    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")

    # Generate C code
    code = hcl.build(s, target)

    return code

# Example usage
if __name__ == "__main__":
    P, Q, R, S, T = 4, 3, 2, 3, 4
    alpha, beta, gamma = 1.0, 1.0, 1.0

    # Generate random matrices
    A = np.random.rand(P, Q)
    B = np.random.rand(Q, R)
    C = np.random.rand(R, S)
    D = np.random.rand(S, T)
    E = np.random.rand(P, T)

    # Compute using HeteroCL
    code = matrix_computation(P, Q, R, S, T, alpha, beta, gamma)
    hcl_result = hcl.asarray(np.zeros((P, T)))
    code(A, B, C, D, E, hcl_result)

    # Compute using NumPy for cross verification
    numpy_result = alpha * np.matmul(np.matmul(A, B), np.matmul(C, D)) + gamma * E

    # Compare results
    assert np.allclose(hcl_result.asnumpy(), numpy_result, rtol=1e-5, atol=1e-5)
    print("HeteroCL result matches NumPy result!")
