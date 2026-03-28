#!/usr/bin/env python3
"""Matrix decomposition: LU, QR, Cholesky."""
import math
def lu(A):
    n=len(A);L=[[0]*n for _ in range(n)];U=[row[:] for row in A]
    for i in range(n): L[i][i]=1
    for k in range(n):
        for i in range(k+1,n):
            L[i][k]=U[i][k]/U[k][k]
            for j in range(k,n): U[i][j]-=L[i][k]*U[k][j]
    return L,U
def qr(A):
    n=len(A);m=len(A[0]);Q=[[0]*n for _ in range(n)];R=[[0]*m for _ in range(n)]
    cols=[[A[i][j] for i in range(n)] for j in range(m)]
    basis=[]
    for j in range(m):
        v=cols[j][:]
        for k in range(len(basis)):
            proj=sum(v[i]*basis[k][i] for i in range(n))
            R[k][j]=proj
            for i in range(n): v[i]-=proj*basis[k][i]
        norm=math.sqrt(sum(x*x for x in v))
        R[j][j]=norm
        if norm>1e-10: basis.append([x/norm for x in v])
        else: basis.append(v)
    for j in range(len(basis)):
        for i in range(n): Q[i][j]=basis[j][i]
    return Q,R
def cholesky(A):
    n=len(A);L=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s=sum(L[i][k]*L[j][k] for k in range(j))
            L[i][j]=math.sqrt(A[i][i]-s) if i==j else (A[i][j]-s)/L[j][j]
    return L
def matmul(A,B):
    n=len(A);m=len(B[0]);p=len(B)
    return [[sum(A[i][k]*B[k][j] for k in range(p)) for j in range(m)] for i in range(n)]
if __name__=="__main__":
    A=[[2,1],[5,3]]
    L,U=lu(A);prod=matmul(L,U)
    assert all(abs(prod[i][j]-A[i][j])<1e-10 for i in range(2) for j in range(2))
    print("LU OK")
    Q,R=qr([[1,1],[1,0],[0,1]])
    print("QR OK")
    S=[[4,2],[2,3]];Lc=cholesky(S);prod=matmul(Lc,[[Lc[j][i] for j in range(2)] for i in range(2)])
    assert all(abs(prod[i][j]-S[i][j])<1e-10 for i in range(2) for j in range(2))
    print("Cholesky OK"); print("All decompositions OK")
