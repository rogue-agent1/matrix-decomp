#!/usr/bin/env python3
"""Matrix decompositions: LU, QR from scratch."""
import sys,math
def lu(A):
    n=len(A);L=[[0]*n for _ in range(n)];U=[row[:] for row in A]
    for i in range(n): L[i][i]=1
    for k in range(n):
        for i in range(k+1,n):
            if U[k][k]==0: continue
            factor=U[i][k]/U[k][k];L[i][k]=factor
            for j in range(k,n): U[i][j]-=factor*U[k][j]
    return L,U
def qr(A):
    m,n=len(A),len(A[0])
    Q=[[A[i][j] for j in range(n)] for i in range(m)]
    R=[[0]*n for _ in range(n)]
    for j in range(n):
        for i in range(j):
            dot=sum(Q[k][i]*Q[k][j] for k in range(m))
            R[i][j]=dot
            for k in range(m): Q[k][j]-=dot*Q[k][i]
        norm=math.sqrt(sum(Q[k][j]**2 for k in range(m)))
        R[j][j]=norm
        if norm>1e-10:
            for k in range(m): Q[k][j]/=norm
    return Q,R
def fmt(M): return '\n'.join('  ['+', '.join(f'{x:7.3f}' for x in row)+']' for row in M)
def main():
    A=[[2,1,1],[4,3,3],[8,7,9]]
    print("A =\n"+fmt(A))
    L,U=lu(A)
    print("\nLU Decomposition:")
    print("L =\n"+fmt(L))
    print("U =\n"+fmt(U))
    Q,R=qr(A)
    print("\nQR Decomposition:")
    print("Q =\n"+fmt(Q))
    print("R =\n"+fmt(R))
if __name__=="__main__": main()
