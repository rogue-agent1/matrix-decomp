#!/usr/bin/env python3
"""matrix_decomp - LU, QR, and Cholesky matrix decomposition."""
import sys, json, math

def mat_mul(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k]*B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

def mat_sub(A, B):
    return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def frobenius(A):
    return math.sqrt(sum(A[i][j]**2 for i in range(len(A)) for j in range(len(A[0]))))

def lu_decompose(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
        for j in range(i, n):
            if i == j: L[i][i] = 1
            else: L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]
    return L, U

def qr_decompose(A):
    n, m = len(A), len(A[0])
    Q = [[A[i][j] for j in range(m)] for i in range(n)]
    R = [[0]*m for _ in range(m)]
    for j in range(m):
        for i in range(j):
            R[i][j] = sum(Q[k][i]*Q[k][j] for k in range(n))
            for k in range(n): Q[k][j] -= R[i][j]*Q[k][i]
        norm = math.sqrt(sum(Q[k][j]**2 for k in range(n)))
        R[j][j] = norm
        if norm > 1e-15:
            for k in range(n): Q[k][j] /= norm
    return Q, R

def cholesky(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j: L[i][j] = math.sqrt(max(A[i][i]-s, 0))
            elif L[j][j] > 1e-15: L[i][j] = (A[i][j]-s)/L[j][j]
    return L

def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def main():
    A = [[4,3,2],[3,5,1],[2,1,6]]
    print("Matrix decomposition demo\n")
    L, U = lu_decompose(A)
    err_lu = frobenius(mat_sub(mat_mul(L,U), A))
    print(f"LU: ||A-LU|| = {err_lu:.2e}")
    Q, R = qr_decompose(A)
    err_qr = frobenius(mat_sub(mat_mul(Q,R), A))
    QtQ = mat_mul(transpose(Q), Q)
    orth = max(abs(QtQ[i][j]-(1 if i==j else 0)) for i in range(3) for j in range(3))
    print(f"QR: ||A-QR|| = {err_qr:.2e}, orthogonality = {orth:.2e}")
    Lc = cholesky(A)
    err_ch = frobenius(mat_sub(mat_mul(Lc, transpose(Lc)), A))
    print(f"Cholesky: ||A-LL'|| = {err_ch:.2e}")

if __name__ == "__main__":
    main()
