import heterocl as hcl
import numpy as np

def top_gemm(P, Q, R, S, T, alpha, beta, gamma, dtype=hcl.Float(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((R, S), "C")
    D = hcl.placeholder((S, T), "D")
    E = hcl.placeholder((P, T), "E")
    F = hcl.placeholder((P, T), "F")

    def kernel_gemm(A, B, C, D, E, F):
        r = hcl.reduce_axis(0, Q, "r")
        s = hcl.reduce_axis(0, S, "s")
        AB = hcl.compute((P, R), lambda x, y: hcl.sum(alpha * A[x, r] * B[r, y], axis=r, dtype=dtype), name="AB")
        CD = hcl.compute((R, T), lambda x, y: hcl.sum(beta * C[x, s] * D[s, y], axis=s, dtype=dtype), name="CD")
        return hcl.compute(F.shape, lambda x, y: AB[x, y] * CD[y, y] + gamma * E[x, y], name="F")

    s = hcl.create_schedule([A, B, C, D, E, F], kernel_gemm)

    #### Applying customizations ####
    AB = kernel_gemm.AB
    CD = kernel_gemm.CD
    F = kernel_gemm.F

    s[AB].compute_at(s[F], F.axis[0])
    s[AB].parallel(AB.axis[0])
    s[CD].compute_at(s[F], F.axis[0])
    s[CD].parallel(CD.axis[0])
    s[F].parallel(F.axis[0])

    return hcl.build(s, target=target)

def main(P=16, Q=22, R=18, S=20, T=24, alpha=0.1, beta=0.1, gamma=0.1, dtype=hcl.Float(32), target=None):
    f = top_gemm(P, Q, R, S, T, alpha, beta, gamma, dtype, target)
    A = np.random.randint(10, size=(P, Q)).astype(np.float32)
    B = np.random.randint(10, size=(Q, R)).astype(np.float32)
    C = np.random.randint(10, size=(R, S)).astype(np.float32)
    D = np.random.randint(10, size=(S, T)).astype(np.float32)
    E = np.random.randint(10, size=(P, T)).astype(np.float32)
    F = np.zeros((P, T), dtype=np.float32)
    f(A, B, C, D, E, F)
    golden = alpha * np.matmul(A, B) * beta * np.matmul(C, D) + gamma * E
    if np.allclose(golden, F):
        print("passed")
    else:
        print("failed")

if __name__ == "__main__":
    main()