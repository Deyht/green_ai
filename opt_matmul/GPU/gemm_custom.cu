#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>

//using namespace nvcuda;

cublasHandle_t handle;


#define BlockDimM 128
#define BlockDimN 128
#define BlockDimK 16

#define WarpDimM 32
#define WarpDimN 16
#define WarpDimK 1

#define ThreadDimM 4
#define ThreadDimN 4


//Custom 2: (Non-Square) Bloc Tiled, use shared memory, and warp accumulate
__global__ void custom_gemm(int TransA, int TransB, 
	int M, int N, int K, 
	float alpha, 
	float *A, int lda, 
	float *B, int ldb, 
	float beta, 
	float *C, int ldc)
{

	int block_row = blockIdx.x;
	int block_col = blockIdx.y;

	float frag_a[ThreadDimM];
	float frag_b[ThreadDimN];

	float accumulator[ThreadDimN][ThreadDimM];

	int c_id = threadIdx.y*blockDim.x + threadIdx.x;
	int warp_id = c_id / 32;
	int thread_id = c_id % 32;
	
	int warp_x = warp_id / (BlockDimM/WarpDimM);
	int warp_y = warp_id % (BlockDimM/WarpDimM);
	
	int frag_x = thread_id / (WarpDimM/ThreadDimM);
	int frag_y = thread_id % (WarpDimM/ThreadDimM);
	
	float *data_block_A;
	float *data_block_B;
	float *data_block_C;
	
	__shared__ float As[BlockDimK][BlockDimM];
	__shared__ float Bs[BlockDimN][BlockDimK];
	
	int l, e_per_th, p_x, p_y;
	
	data_block_C = &C[block_col*BlockDimN*M + block_row*BlockDimM];
	
	#pragma unroll
	for(int thread_x = 0; thread_x < ThreadDimN; thread_x ++)
		#pragma unroll
		for(int thread_y = 0; thread_y < ThreadDimM; thread_y++)
			accumulator[thread_x][thread_y] = 0.0f;
	
	for(int block_k = 0; block_k < (K/BlockDimK); block_k++)
	{
		data_block_A = &A[block_k*BlockDimK*M + block_row*BlockDimM];
		data_block_B = &B[block_col*BlockDimN*K + block_k*BlockDimK];
		
		e_per_th = (BlockDimM*BlockDimK)/(blockDim.x*blockDim.y);
		for(l = 0; l < e_per_th; l++)
		{
			p_x = (c_id*e_per_th + l)/BlockDimM;
			p_y = (c_id*e_per_th + l)%BlockDimM;
			
			As[p_x][p_y] = data_block_A[p_x*M + p_y];
		}
		
		__syncthreads();
		
		e_per_th = (BlockDimN*BlockDimK)/(blockDim.x*blockDim.y);
		for(l = 0; l < e_per_th; l++)
		{
			p_x = (c_id*e_per_th + l)/BlockDimK;
			p_y = (c_id*e_per_th + l)%BlockDimK;
			
			Bs[p_x][p_y] = data_block_B[p_x*K + p_y];
		}
		
		__syncthreads();
		
		#pragma unroll
		for(int warp_k = 0; warp_k < BlockDimK; warp_k++)
		{
			#pragma unroll
			for(int thread_y = 0; thread_y < ThreadDimM; thread_y++)
				frag_a[thread_y] = As[warp_k][warp_y*WarpDimM + frag_y*ThreadDimM + thread_y];
			#pragma unroll
			for(int thread_x = 0; thread_x < ThreadDimN; thread_x ++)
				frag_b[thread_x] = Bs[warp_x*WarpDimN + frag_x*ThreadDimN + thread_x][warp_k];
			
			#pragma unroll
			for(int thread_x = 0; thread_x < ThreadDimN; thread_x ++)
				#pragma unroll
				for(int thread_y = 0; thread_y < ThreadDimM; thread_y++)
					accumulator[thread_x][thread_y] +=
						frag_a[thread_y]*frag_b[thread_x];
		}
		__syncthreads();
	}
	
	#pragma unroll
	for(int thread_x = 0; thread_x < ThreadDimN; thread_x ++)
	{
		
		#pragma unroll
		for(int thread_y = 0; thread_y < ThreadDimM; thread_y++)
		
			data_block_C[(warp_x*WarpDimN + frag_x*ThreadDimN + thread_x)*M + (warp_y*WarpDimM + frag_y*ThreadDimM + thread_y)] = alpha*accumulator[thread_x][thread_y] + beta*data_block_C[(warp_x*WarpDimN + frag_x*ThreadDimN + thread_x)*M + (warp_y*WarpDimM + frag_y*ThreadDimM + thread_y)] ;
	}
	
}



int main()
{
	size_t i;

	size_t M = 4096, N = 4096, K = 4096;
	float *A, *B, *C, *C_bis;
	float *cu_A, *cu_B, *cu_C;

	float alpha = 1.0f, beta = 0.0f;

	printf("M:%ld N:%ld K:%ld\n", M, K, K);

	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	A = (float*) calloc(M*K,sizeof(float));
	B = (float*) calloc(K*N,sizeof(float));
	C = (float*) calloc(M*N,sizeof(float));
	C_bis = (float*) calloc(M*N,sizeof(float));

	for(i = 0; i < M*K; i++)
		A[i] = ((float)rand()/RAND_MAX-0.5)*0.1;
	
	for(i = 0; i < K*N; i++)
		B[i] = ((float)rand()/RAND_MAX-0.5)*0.1;

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

	//CUSTOM FP32 VERSION

	cudaMemcpy(cu_C, C_bis, M*N*sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(BlockDimM/ThreadDimM, BlockDimN/ThreadDimN);
	
	dim3 numBlocks(M/BlockDimM, N/BlockDimN);

	cudaEventRecord(start);
	custom_gemm<<< numBlocks, threadsPerBlock >>>(0, 0, M, N, K, alpha, cu_A, M, cu_B, K, beta, cu_C, M);
	cudaEventRecord(stop);
	
	cudaMemcpy(C_bis, cu_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Custom FP32 gemm: %f\nElapsed time: %f ms\n", C_bis[M*N/2], milliseconds);
	
	for (i = 0; i < M*N; i++)
		if((C[i] - C_bis[i])*(C[i] - C_bis[i]) > 0.00001)
		{
			printf("ERROR ! MATRIX DIFF !\n");
			printf("i:%ld C:%f C_bis:%f\n", i, C[i], C_bis[i]);
			exit(EXIT_FAILURE);
		}
	
	exit(EXIT_SUCCESS);
}

        
        
        
        
        

