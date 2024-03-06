nvcc -O3 gemm_custom.cu -o gemm_custom -arch=sm_75 -lcublas -lcudart
