#!/usr/bin/env python3
"""matrix_decomp - Matrix decompositions (LU, Cholesky, QR)."""
import sys, math

def lu_decompose(A):
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for k in range(i, n):
            s = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - s
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                s = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - s) / U[i][i]
    return L, U

def cholesky(A):
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def mat_mul(A, B):
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    return C

def test():
    A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    L, U = lu_decompose(A)
    # L * U should equal A
    product = mat_mul(L, U)
    for i in range(3):
        for j in range(3):
            assert abs(product[i][j] - A[i][j]) < 1e-9
    # Cholesky: A must be positive definite
    spd = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    Lc = cholesky(spd)
    # L * L^T should equal A
    Lt = [[Lc[j][i] for j in range(3)] for i in range(3)]
    product2 = mat_mul(Lc, Lt)
    for i in range(3):
        for j in range(3):
            assert abs(product2[i][j] - spd[i][j]) < 1e-9
    print("OK: matrix_decomp")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print("Usage: matrix_decomp.py test")
