import heterocl as hcl
import numpy as np

def compute_kernel_naive(A, B, C, D, E, alpha, beta, gamma):
    # Create HeteroCL placeholders for matrices
    P, Q, R, S, T = A.shape[0], B.shape[1], B.shape[0], C.shape[1], D.shape[1]
    AB = hcl.placeholder((P, R), name="AB")
    CD = hcl.placeholder((R, T), name="CD")
    E_hcl = hcl.placeholder((P, T), name="E")

    # Create HeteroCL compute kernel
    def compute_kernel(F):
        with hcl.Stage():
            F[0, 0] = alpha * AB[0, 0] * beta * CD[0, 0] + gamma * E_hcl[0, 0]

    # Compile the HeteroCL kernel
    s = hcl.create_schedule([AB, CD, E_hcl], compute_kernel)
    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    f = hcl.build(s, target=target)

    # Execute the kernel
    AB_np = np.dot(A, B)
    CD_np = np.dot(C, D)
    E_np = E
    F_np = np.zeros((P, T))
    f(AB_np, CD_np, E_np, F_np)

    return F_np

# Example usage
P, Q, R, S, T = 3, 4, 5, 6, 7  # Example dimensions
A = np.random.rand(P, Q)
B = np.random.rand(Q, R)
C = np.random.rand(R, S)
D = np.random.rand(S, T)
E = np.random.rand(P, T)
alpha, beta, gamma = 1.0, 2.0, 0.5

result = compute_kernel_naive(A, B, C, D, E, alpha, beta, gamma)
print("Result (F):\n", result)