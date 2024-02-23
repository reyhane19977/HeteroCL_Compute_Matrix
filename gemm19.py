import heterocl as hcl
import numpy as np

def top_gemm(P, Q, R, S, T, alpha, beta, gamma, dtype=hcl.Float(32), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((R, S), "C")
    D = hcl.placeholder((S, T), "D")
    E = hcl.placeholder((P, T), "E")

    def kernel_gemm(A, B, C, D, E):
        r_AB = hcl.reduce_axis(0, Q, "r_AB")
        r_CD = hcl.reduce_axis(0, S, "r_CD")
        out_AB = hcl.compute(
            (P, R),
            lambda x, y: hcl.sum(alpha * A[x, r_AB] * B[r_AB, y], axis=r_AB, dtype=dtype),
            name="out_AB",
        )
        out_CD = hcl.compute(
            (R, T),
            lambda x, y: hcl.sum(beta * C[x, r_CD] * D[r_CD, y], axis=r_CD, dtype=dtype),
            name="out_CD",
        )
        r_F = hcl.reduce_axis(0, R, "r_F")
        return hcl.compute(
            E.shape, lambda x, y: hcl.sum(out_AB[x, r_F] * out_CD[r_F, y], axis=r_F, dtype=dtype) + gamma * E[x, y], name="F"
        )

    s = hcl.create_schedule([A, B, C, D, E], kernel_gemm)
    F = kernel_gemm.F
    s[F].parallel(F.axis[0])
    return hcl.build(s, target=target)

def main(P=16, Q=22, R=18, S=20, T=24, alpha=0.1, beta=0.1, gamma=0.1, dtype=hcl.Float(32), target=None):
    print("Executing main function...")
    f = top_gemm(P, Q, R, S, T, alpha, beta, gamma, dtype, target)
    A = hcl.asarray(np.random.rand(P, Q), dtype=dtype)
    B = hcl.asarray(np.random.rand(Q, R), dtype=dtype)
    C = hcl.asarray(np.random.rand(R, S), dtype=dtype)
    D = hcl.asarray(np.random.rand(S, T), dtype=dtype)
    E = hcl.asarray(np.random.rand(P, T), dtype=dtype)
    F = hcl.asarray(np.zeros((P, T)), dtype=dtype)
    f(A, B, C, D, E, F)
    golden = alpha * np.matmul(A.asnumpy(), B.asnumpy()) @ (beta * np.matmul(C.asnumpy(), D.asnumpy())) + gamma * E.asnumpy()
    
    print("Golden matrix:\n", golden)
    print("Computed matrix F:\n", F.asnumpy())

    if np.allclose(golden, F.asnumpy()):
        print("passed")
    else:
        print("failed")