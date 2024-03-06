#include <stdio.h>
#include <stdlib.h>

#include <mma.h>

#include <cuda_fp16.h>
#include <cublas_v2.h>

using namespace nvcuda;

cublasHandle_t handle;


//Example of a wmma kernel operation
__global__ void custom_wmma(half *A, half *B, float *C)
{

	// Declare the fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	// Initialize the output to zero
	wmma::fill_fragment(c_frag, 0.0f);

	// Load the inputs
	wmma::load_matrix_sync(a_frag, A, 16);
	wmma::load_matrix_sync(b_frag, B, 16);

	// Perform the matrix multiplication
	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

	// Store the output
	wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_col_major);
}



int main()
{
	long int i;

	long int M = 4096, N = 4096, K = 4096;
	float *A, *B, *C;
	float *cu_A, *cu_B, *cu_C;

	float alpha = 1.0f, beta = 0.0f;
	float cu_alpha = 1.0f, cu_beta = 0.0f;
	half cu_h_alpha = 1.0f, cu_h_beta = 0.0f;

	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("M:%ld N:%ld K:%ld\n", M, K, K);

	A = (float*) calloc(M*K,sizeof(float));
	B = (float*) calloc(K*N,sizeof(float));
	C = (float*) calloc(M*N,sizeof(float));

	for(i = 0; i < M*K; i++)
		A[i] = ((float)rand()/RAND_MAX-0.5)*0.1;
	
	for(i = 0; i < K*N; i++)
		B[i] = ((float)rand()/RAND_MAX-0.5)*0.1;

	half *hA, *hB, *hC;
	half *cu_hA, *cu_hB, *cu_hC;

	hA = (half*) calloc(M*K, sizeof(half));
	hB = (half*) calloc(K*N, sizeof(half));
	hC = (half*) calloc(M*N, sizeof(half));

	for(i = 0; i < M*K; i++)
		hA[i] = (half)(A[i]);
	for(i = 0; i < K*N; i++)
		hB[i] = (half)(B[i]);

	//cuBLAS FP32 VERSION

	cublasStatus_t cublasStat = cublasCreate(&handle);
	
	cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
	
	cudaMalloc(&cu_A, M*K*sizeof(float));
	cudaMalloc(&cu_B, K*N*sizeof(float));
	cudaMalloc(&cu_C, M*N*sizeof(float));

	cudaMemcpy(cu_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cu_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cu_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, cu_A, M, cu_B, K, &beta, cu_C, M);
	cudaEventRecord(stop);

	cudaMemcpy(C, cu_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	
	printf("Regular FP32 gemm: %f\nElapsed time: %f ms\n", C[M*N/2], milliseconds);


	//TENSOR CORE MIXED PRECISION VERSION
	cublasStat = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

	cudaMalloc(&cu_hA, M*K*sizeof(half));
	cudaMalloc(&cu_hB, K*N*sizeof(half));
	cudaMalloc(&cu_hC, M*N*sizeof(half));

	cudaMemcpy(cu_hA, hA, M*K*sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(cu_hB, hB, K*N*sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(cu_hC, hC, M*N*sizeof(half), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &cu_alpha,
		          cu_hA, CUDA_R_16F, M, cu_hB, CUDA_R_16F, K,
		          &cu_beta, cu_hC, CUDA_R_16F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	cudaEventRecord(stop);

	cudaMemcpy(hC, cu_hC, M*N*sizeof(half), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	
	printf("TC Mixed FP16: %f\nElapsed time: %f ms\n", (float)(hC[M*N/2]), milliseconds);

	

	exit(EXIT_SUCCESS);
}

        
        
        
        
        

