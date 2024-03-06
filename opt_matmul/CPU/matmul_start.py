import numpy as np

def matmul_naive(A, B, M, N, K):
	# Add your own naive 3-loop implementation here
	return

M = 256
N = 256
K = 256

np.random.seed(0)

A = (np.random.random((M,K))-0.5)*0.1
B = (np.random.random((K,N))-0.5)-0.1

C = matmul_naive(A,B,M,N,K)

print (C[M//2,N//2])

