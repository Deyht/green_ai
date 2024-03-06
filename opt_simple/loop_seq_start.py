import numpy as np


def loop_naive(v_size, rad, A, B):
	# Add your own naive 6-loop implementation here
	return




v_size = 64
rad = 3

# Initialize A with values using broadcasting
x, y, z = np.ogrid[:v_size, :v_size, :v_size]
A = (x + y + z).astype("float32")

# Initialize B as an empty array with the same shape as A
B = np.zeros_like(A).astype("float32")
			
loop_naiv(v_size,rad,A,B)

print(B[v_size//2, v_size//2, v_size//2])





