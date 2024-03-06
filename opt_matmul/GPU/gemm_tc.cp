nvcc -O3 gemm_tc.cu -o gemm_tc -arch=sm_75 -lcublas -lcudart
