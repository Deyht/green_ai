gcc -O3 -fopenmp -march=native -ffast-math -funroll-loops matmul_para.c -o matmul_para -lm -lopenblas
