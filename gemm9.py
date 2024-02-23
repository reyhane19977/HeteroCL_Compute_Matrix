import heterocl as hcl
import numpy as np
import time

def compute_kernel(A, B, C, D, E, alpha, beta, gamma):
    P, Q, R, S, T = A.shape[0], A.shape[1], B.shape[1], C.shape[1], D.shape[1]
    def kernel(A, B, C, D, E, F):
        with hcl.Stage() as s:
            A_ = hcl.placeholder((P, Q), dtype=hcl.Float())
            B_ = hcl.placeholder((Q, R), dtype=hcl.Float())
            C_ = hcl.placeholder((R, S), dtype=hcl.Float())
            D_ = hcl.placeholder((S, T), dtype=hcl.Float())
            E_ = hcl.placeholder((P, T), dtype=hcl.Float())

            F_ = hcl.compute((P, T), lambda i, j: alpha * hcl.sum(A_[i, k] * B_[k, j], axis=0, dtype=hcl.Float()) * beta * hcl.sum(C_[k, l] * D_[l, j], axis=0, dtype=hcl.Float()) + gamma * E_[i, j], dtype=hcl.Float(), name="F")

            return s

    hcl_A = hcl.placeholder((P, Q), dtype=hcl.Float())
    hcl_B = hcl.placeholder((Q, R), dtype=hcl.Float())
    hcl_C = hcl.placeholder((R, S), dtype=hcl.Float())
    hcl_D = hcl.placeholder((S, T), dtype=hcl.Float())
    hcl_E = hcl.placeholder((P, T), dtype=hcl.Float())
    hcl_F = hcl.placeholder((P, T), dtype=hcl.Float())
    
    s = kernel(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)
    target = hcl.platform.zc706
    s = hcl.create_schedule([hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F], s)
    s.to([hcl_A, hcl_B, hcl_C, hcl_D, hcl_E], target.xcel)
    s.to(hcl_F, target.host)
    f = hcl.build(s, target=target)
    
    return f

def compute_kernel_optimized(A, B, C, D, E, alpha, beta, gamma):
    pass

# Generate random matrices with specified dimensions
P, Q, R, S, T = 10, 20, 30, 40, 50
alpha, beta, gamma = 1.0, 2.0, 3.0
A = np.random.rand(P, Q)
B = np.random.rand(Q, R)
C = np.random.rand(R, S)
D = np.random.rand(S, T)
E = np.random.rand(P, T)
F = np.zeros((P, T))

# Naive computation
start_time = time.time()
f_naive = compute_kernel(A, B, C, D, E, alpha, beta, gamma)
naive_result = f_naive(A, B, C, D, E, F)
naive_time = time.time() - start_time

print("Naive computation time:", naive_time)
print("Optimized computation time:", optimized_time)
