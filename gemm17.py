import heterocl as hcl
import numpy as np

def gemm_compute_kernel(P, Q, R, S, T, alpha, beta, gamma, A, B, C, D, E):
    # Create a placeholder for the result
    F = hcl.placeholder((P, T), "F")

    # Define the compute
    def kernel(A, B, C, D, E, F):
        rA = hcl.reduce_axis(0, Q, "rA")
        rB = hcl.reduce_axis(0, S, "rB")
        F[P, T] = hcl.sum(alpha * A[P, rA] * B[rA, R] * beta * C[R, rB] * D[rB, T], axis=[rA, rB], name="F_sum") + gamma * E[P, T]

    # Create the schedule and the function
    s = hcl.create_schedule([A, B, C, D, E, F], kernel)
    f = hcl.build(s)

    return f

# Define the testbench
def testbench():
    alpha, beta, gamma = 1.0, 2.0, 3.0
    P, Q, R, S, T = 10, 20, 20, 30, 10  # Example dimensions
    
    # Initialize matrices with random values
    matrix_A = np.random.rand(P, Q).astype(np.float32)
    matrix_B = np.random.rand(Q, R).astype(np.float32)
    matrix_C = np.random.rand(R, S).astype(np.float32)
    matrix_D = np.random.rand(S, T).astype(np.float32)
    matrix_E = np.random.rand(P, T).astype(np.float32)
    matrix_F = np.zeros((P, T)).astype(np.float32)

    hcl_A = hcl.asarray(matrix_A)
    hcl_B = hcl.asarray(matrix_B)
    hcl_C = hcl.asarray(matrix_C)
    hcl_D = hcl.asarray(matrix_D)
    hcl_E = hcl.asarray(matrix_E)
    hcl_F = hcl.asarray(matrix_F)

    # Get the compute kernel
    f = gemm_compute_kernel(P, Q, R, S, T, alpha, beta, gamma, hcl_A, hcl_B, hcl_C, hcl_D, hcl_E)
    
    # Execute the kernel
    f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)

    # Get back the result
    np_F = hcl_F.asnumpy()
    print("Result F:", np_F)

testbench()  # Call the test bench to execute