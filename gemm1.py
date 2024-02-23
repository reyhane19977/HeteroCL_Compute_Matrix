import heterocl as hcl
import numpy as np
import time

def matrix_multiply_optimized():
    # Define the dimensions
    P, Q, R, S, T = 10, 10, 10, 10, 10

    # Define the matrices
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((R, S), "C")
    D = hcl.placeholder((S, T), "D")
    E = hcl.placeholder((P, T), "E")

    # Define the constants
    alpha = hcl.placeholder((), "alpha")
    beta = hcl.placeholder((), "beta")
    gamma = hcl.placeholder((), "gamma")

    def kernel(A, B, C, D, E, alpha, beta, gamma):
        # Define the reduction function for matrix multiplication
        def matmul_reduce(i, j, k):
            return hcl.sum(A[i, k] * B[k, j], axis=k, dtype=hcl.Float())

        # Intermediate matrices
        AB = hcl.compute((P, R), lambda i, j: matmul_reduce(i, j, hcl.reduce_axis(0, Q)), "AB")
        CD = hcl.compute((R, T), lambda i, j: matmul_reduce(i, j, hcl.reduce_axis(0, S)), "CD")

        # Final matrix
        F = hcl.compute((P, T), lambda i, j: alpha[0] * AB[i, j] * beta[0] * CD[j, j] + gamma[0] * E[i, j], "F")
        return F, AB, CD

    # Create a HeteroCL schedule
    s = hcl.create_schedule([A, B, C, D, E, alpha, beta, gamma], kernel)

    # Get the intermediate matrices from the kernel
    _, AB, CD = kernel(A, B, C, D, E, alpha, beta, gamma)

    # Apply optimizations (e.g., loop unrolling, array partitioning)
    s[AB].unroll(AB.axis[1])
    s[CD].unroll(CD.axis[1])
    s[F].unroll(F.axis[1])

    # Build the optimized kernel
    f_opt = hcl.build(s)

    # Create random input matrices
    hcl_A = hcl.asarray(np.random.rand(P, Q))
    hcl_B = hcl.asarray(np.random.rand(Q, R))
    hcl_C = hcl.asarray(np.random.rand(R, S))
    hcl_D = hcl.asarray(np.random.rand(S, T))
    hcl_E = hcl.asarray(np.random.rand(P, T))

    # Constants
    hcl_alpha = hcl.asarray(np.array([1.0]))
    hcl_beta = hcl.asarray(np.array([1.0]))
    hcl_gamma = hcl.asarray(np.array([1.0]))

    # Output matrix
    hcl_F = hcl.asarray(np.zeros((P, T)))

    # Execute the optimized kernel
    start_time = time.time()
    f_opt(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_alpha, hcl_beta, hcl_gamma, hcl_F)
    end_time = time.time()

    # Print the execution time
    print("Optimized execution time: {:.6f} seconds".format(end_time - start_time))

    # Return the result
    return hcl_F.asnumpy()

# Test the optimized implementation
F_optimized = matrix_multiply_optimized()
print("Optimized result:\n", F_optimized)


import heterocl as hcl
import numpy as np

def top_compute(P, Q, R, S, T, alpha, beta, gamma, dtype=hcl.Float(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((R, S), "C")
    D = hcl.placeholder((S, T), "D")
    E = hcl.placeholder((P, T), "E")
    F = hcl.placeholder((P, T), "F")

    def kernel_compute(A, B, C, D, E, F):
        r = hcl.reduce_axis(0, Q, "r")
        s = hcl.reduce_axis(0, S, "s")
        out_AB = hcl.compute(
            (P, R),
            lambda x, y: hcl.sum(A[x, r] * B[r, y], axis=r),
            name="out_AB",
        )
        out_CD = hcl.compute(
            (R, T),
            lambda x, y: hcl.sum(C[x, s] * D[s, y], axis=s),
            name="out_CD",
        )
        return hcl.compute(
            F.shape,
            lambda x, y: alpha * out_AB[x, y] * beta * out_CD[x, y] + gamma * E[x, y],
            name="F",
        )

    s = hcl.create_schedule([A, B, C, D, E, F], kernel_compute)
    return hcl.build(s, target=target)

def main(P=8, Q=8, R=8, S=8, T=8, alpha=1.0, beta=1.0, gamma=1.0, dtype=hcl.Float(32), target=None):
    f = top_compute(P, Q, R, S, T, alpha, beta, gamma, dtype, target)
    A = np.random.rand(P, Q).astype(np.float32)
    B = np.random.rand(Q, R).astype(np.float32)
    C = np.random.rand(R, S).astype(np.float32)
    D = np.random.rand(S, T).astype(np.float32)
    E = np.random.rand(P, T).astype(np.float32)
    F = np.zeros((P, T), dtype=np.float32)
    f(A, B, C, D, E, F)
    golden = alpha * np.matmul(A, np.matmul(B, C)) * beta * np.matmul(C, D) + gamma * E
    if np.allclose(golden, F):
        print("passed")
    else:
        print("failed")

if __name__ == "__main__":
    main()
