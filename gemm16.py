import heterocl as hcl
import numpy as np

def gemm_compute_kernel(alpha, beta, gamma, A_shape, B_shape, C_shape, D_shape, E_shape):
    P, Q = A_shape
    _, R = B_shape
    _, S = C_shape
    _, T = E_shape

    # Create placeholders for the matrices
    A = hcl.placeholder(A_shape, "A")
    B = hcl.placeholder(B_shape, "B")
    C = hcl.placeholder(C_shape, "C")
    D = hcl.placeholder(D_shape, "D")
    E = hcl.placeholder(E_shape, "E")

    # Define the compute kernel
    def kernel(A, B, C, D, E):
        rA = hcl.reduce_axis(0, Q, "rA")
        rB = hcl.reduce_axis(0, S, "rB")
        return hcl.compute(
            (P, T),
            lambda i, j: hcl.sum(
                alpha * A[i, rA] * B[rA, j % R] * beta * C[i % P, rB] * D[rB, j], 
                axis=[rA, rB],
                dtype=hcl.Float()),
            name="F",
        ) + gamma * E

    # Create the schedule
    s = hcl.create_schedule([A, B, C, D, E], kernel)

    # Build the kernel
    return hcl.build(s)

def testbench():
    alpha, beta, gamma = 1.0, 2.0, 3.0
    A_shape = (10, 20)
    B_shape = (20, 20)
    C_shape = (10, 30)
    D_shape = (30, 10)
    E_shape = (10, 10)

    # Initialize matrices with random values
    matrix_A = np.random.rand(*A_shape).astype(np.float32)
    matrix_B = np.random.rand(*B_shape).astype(np.float32)
    matrix_C = np.random.rand(*C_shape).astype(np.float32)
    matrix_D = np.random.rand(*D_shape).astype(np.float32)
    matrix_E = np.random.rand(*E_shape).astype(np.float32)
    matrix_F = np.zeros((A_shape[0], D_shape[1])).astype(np.float32)

    hcl_A = hcl.asarray(matrix_A)
    hcl_B = hcl.asarray(matrix_B)
    hcl_C = hcl.asarray(matrix_C)
    hcl_D = hcl.asarray(matrix_D)
    hcl_E = hcl.asarray(matrix_E)
    hcl_F = hcl.asarray(matrix_F)

    # Get the compute kernel
    f = gemm_compute_kernel(alpha, beta, gamma, A_shape, B_shape, C_shape, D_shape, E_shape)
    
    # Execute the kernel
    f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)

    # Get back the result
    np_F = hcl_F.asnumpy()
    print("Result F:", np_F)

testbench()  # Call the test bench to execute