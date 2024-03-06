
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>

#include <sys/time.h>

struct timeval timer;

void init_timing(struct timeval* tstart)
{
	gettimeofday(tstart, NULL);
}

float elapsed_time(struct timeval tstart)
{
	struct timeval tmp;
	long long diff;
	gettimeofday(&tmp, NULL);
	diff = tmp.tv_usec - tstart.tv_usec;
	diff += (tmp.tv_sec - tstart.tv_sec) * 1000000;
	return ((float)diff*1.0e-6);
}




//Naive triple loop matmul
void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K)
{
	int i,j,k;
	double acc;
	
	for(i = 0; i < N; i++)
		for(j = 0; j < M; j++)	
			for(k = 0; k < K; k++)
				C[i*M+j] += A[k*M+j] * B[i*K+k];
}


//v1 + simple register accumulate
void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K)
{
	int i,j,k;
	float acc;
	
	for(i = 0; i < N; i++)
		for(j = 0; j < M; j++)	
		{	
			acc = 0.0;
			for(k = 0; k < K; k++)
				acc += A[k*M+j] * B[i*K+k];
			C[i*M+j] = acc;
		}
}

//v2 + transposed A // Try it with O3 and O2 to see the effect of auto vectorize !
void matmul_v3(const float *A, const float *B, float *C, int M, int N, int K)
{
	int i,j,k;
	float *t_A = (float*) malloc(M*K*sizeof(float));
	float acc;
	
	for(j = 0; j < M; j++)	
		for(i = 0; i < K; i++)
			t_A[j*K+i] = A[i*M+j];
	
	for(i = 0; i < N; i++)
		for(j = 0; j < M; j++)
		{	
			acc = 0.0;
			for(k = 0; k < K; k++)
				acc += t_A[j*K+k] * B[i*K+k];
			C[i*M+j] = acc;
		}
}

//v3 + manual vectorization through memory aligned SIMD operations 
//Note, require C11 ISO, not compatible with C99 while auto vectorization is compatible
//Try it with O2 to see that it preserves the level of performance of O3

// Define a memory aligned vector type of 8 floats (32 bit)
typedef float vec __attribute__ (( vector_size(32) ));

void matmul_v4(const float *A, const float *B, float *C, int M, int N, int K)
{
	if(K % 8 != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}
	
	//How to declare a vector dynamically, composed of X 32 bit elements ?
	//vec *a = (vec*) aligned_alloc(32,X*32);  
	
	//How to access a vector sub element ?
	// a[i/8][i%8] = A[i];
	
	//How to allocated and initialize a vector to a constant value
	//vec acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	
	//1° Create the vectorized versions of A and B
	
	//2° Fill the vectorized versions
	
	//3° Do the i,j,k loops with the vectorized accumulate loop over k
	
}



//Matmul using FMA operation on a sub-part of C and optimizing the cache usage
//No need for transposition in this version as most of the work is done in cache anyway

#define ker_w 6
#define ker_h 16

void kernel(const float *_a, const float *_b, float *_c, 
	int M, int N, int K, int i, int j, int k_start, int k_stop) 
{
	int k, b_m, b_n, l;
	float val;
	//declared in cache
	vec t[ker_h/8][ker_w] = {};

	//1° Loop over k that do the vectorized accumulate inside the kernel region using an implicit fma instruction

	//2° accumulate the result in the corresponding region of C
}


void matmul_v5(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j,k;
	
	if(M % ker_w != 0 || N % ker_h != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	//Simple loop over M and N kernel regions that call the kernel

}



//V4 but with blocking decomposition to successively load sub parts of A and B
//into L2 and L1 cache for maximum reuse. Maximizing the achievable memory bandwith

void matmul_v6(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j,k;
	
	// Set the size of the blocks and order them by importance into the different cache levels
	const int l3 = X; //Number of rows from A
	const int l2 = X; //Number of columns from B
	const int l1 = X; //Number of columns from A
	
	if(M % ker_w != 0 || N % ker_h != 0 || l2 % ker_w != 0 || l3 % ker_h != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	//5-nested loop version that goes through the blocks in cache and then call the kernel on the sub regions in the current block
}


int main()
{
	size_t i;
	int scale = 40;
	size_t M = 48*scale, N = 48*scale, K = 48*scale;
	float *A, *B, *C, *C_bis;

	float alpha = 1.0f, beta = 0.0f;

	printf("M:%ld N:%ld K:%ld\n", M, N, K);

	A = (float*) calloc(M*K,sizeof(float));
	B = (float*) calloc(K*N,sizeof(float));
	C = (float*) calloc(M*N,sizeof(float));
	C_bis = (float*) calloc(M*N,sizeof(float));

	for(i = 0; i < M*K; i++)
		A[i] = ((float)rand()/RAND_MAX-0.5)*0.1;
	
	for(i = 0; i < K*N; i++)
		B[i] = ((float)rand()/RAND_MAX-0.5)*0.1;

	init_timing(&timer);

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		M, N, K, alpha, A, M, B, K, beta, C, M);

	printf("Regular sgemm: %f - Elapsed time: %f s\n", C[M*N/2], elapsed_time(timer));

	init_timing(&timer);

	matmul_v3(A,B,C_bis,M,N,K);
	
	printf("Custom sgemm: %f - Elapsed time: %f s\n", C_bis[M*N/2], elapsed_time(timer));
	
	for (i = 0; i < M*N; i++)
		if((C[i] - C_bis[i])*(C[i] - C_bis[i]) > 0.00001)
		{
			printf("ERROR ! MATRIX DIFF !\n");
			printf("i:%ld C:%f C_bis:%f\n", i, C[i], C_bis[i]);
			exit(EXIT_FAILURE);
		}
	exit(EXIT_SUCCESS);
}

		
		
