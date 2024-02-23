import heterocl as hcl
import numpy as np
import time

def compute_kernel(alpha, beta, gamma):
    P, Q, R, S, T = 10, 10, 10, 10, 10  # Example dimensions
    A = hcl.placeholder((P, Q), name="A")
    B = hcl.placeholder((Q, R), name="B")
    C = hcl.placeholder((R, S), name="C")
    D = hcl.placeholder((S, T), name="D")
    E = hcl.placeholder((P, T), name="E")
    F = hcl.placeholder((P, T), name="F")

    def kernel(A, B, C, D, E, F):
        def compute_f():
            res = hcl.compute((P, T), lambda x, y: alpha * sum(A[x, k] * B[k, y] for k in range(Q)) * beta *
                              sum(C[k, y] * D[k, z] for k in range(R) for z in range(T)) +
                              gamma * E[x, y])
            return res

        return hcl.update(F, lambda x, y: compute_f()[x, y])

    s = hcl.create_schedule([A, B, C, D, E, F], kernel)
    target = hcl.platform.zc706
    s.to([A, B, C, D, E], target.xcel)
    s.to(F, target.host)

    return hcl.build(s)

# Test bench
def test(alpha, beta, gamma):
    hcl_init = compute_kernel(alpha, beta, gamma)
    hcl_array_A = hcl.asarray(np.random.rand(10, 10))
    hcl_array_B = hcl.asarray(np.random.rand(10, 10))
    hcl_array_C = hcl.asarray(np.random.rand(10, 10))
    hcl_array_D = hcl.asarray(np.random.rand(10, 10))
    hcl_array_E = hcl.asarray(np.random.rand(10, 10))
    hcl_array_F = hcl.asarray(np.zeros((10, 10)))

    start_time = time.time()
    hcl_init(hcl_array_A, hcl_array_B, hcl_array_C, hcl_array_D, hcl_array_E, hcl_array_F)
    execution_time = time.time() - start_time

    result_F = hcl_array_F.asnumpy()
    # Print or verify result
    print("Execution time:", execution_time)
    print("Result F:", result_F)

# Run test bench
test(0.5, 0.5, 0.5)