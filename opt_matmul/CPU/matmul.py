import numpy as np
from numba import jit, prange

#@jit(nopython=True, cache=True, fastmath=False)
def matmul_naive(A, B, M, N, K):
	C = np.zeros((M,N))
	for i in range(0,M):
		for j in range(0,N):
			for k in range(0,K):
				C[i,j] += A[i,k] * B[k,j]
	return C


M = 256
N = 256
K = 256

np.random.seed(0)

A = (np.random.random((M,K))-0.5)*0.1
B = (np.random.random((K,N))-0.5)-0.1

#Select one
#C = A@B
#C = np.matmul(A,B)
#C = np.dot(A,B)
C = matmul_naive(A,B,M,N,K)



print (C[M//2,N//2])

