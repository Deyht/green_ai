
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
	int i,j,k;
	// Rounded number of 8-element vec in a K
	if(k % 8 != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}
	
	int n_K = K/8;
	
	
	vec *a = (vec*) aligned_alloc(32,M*n_K*32);
	vec *b = (vec*) aligned_alloc(32,N*n_K*32);
	
	for(j = 0; j < M; j++)	
		for(i = 0; i < K; i++)
			a[j*n_K + i/8][i%8] = A[i*M+j];
	
	for(j = 0; j < N; j++)	
		for(i = 0; i < K; i++)
			b[j*n_K + i/8][i%8] = B[j*K+i];
	
	for(i = 0; i < N; i++)
		for(j = 0; j < M; j++)
		{
			vec acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			for(k = 0; k < n_K; k++)
				acc += a[j*n_K+k] * b[i*n_K+k];
			
			for(k = 0; k < 8; k++)
				C[i*M+j] += acc[k];
		}
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

	for(k = k_start; k < k_stop; k++)
	{
		for(b_n = 0; b_n < ker_w; b_n++) 
	   	{
			//brodcast B value to the full vector
			val = _b[(j+b_n)*K+k];
			vec beta = {val, val, val, val, val, val, val, val};
			
			for(b_m = 0; b_m < ker_h/8; b_m++)
			{
				vec alpha;
				for(l = 0; l < 8; l++)
					alpha[l] = _a[i+k*M+b_m*8+l];
				t[b_m][b_n] += alpha * beta; // converts to an fma
			}
		}
	}

	// write the results back to C
	for(b_n = 0; b_n < ker_w; b_n++)
		for(b_m = 0; b_m < ker_h/8; b_m++)
			for(l = 0; l < 8; l++)
				_c[i+(j+b_n)*M+b_m*8+l] += t[b_m][b_n][l];
}


void matmul_v5(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j,k;
	
	if(M % ker_w != 0 || N % ker_h != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	for(j = 0; j < N; j += ker_w)
		for(i = 0; i < M; i += ker_h)
			kernel(A, B, C, M, N, K, i, j, 0, K);

}



//V4 but with blocking decomposition to successively load sub parts of A and B
//into L2 and L1 cache for maximum reuse. Maximizing the achievable memory bandwith

void matmul_v6(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j,k;
	
	const int l3 = 64; //Number of rows from A
	const int l2 = 120; //Number of columns from B
	const int l1 = 240; //Number of columns from A
	
	if(M % ker_w != 0 || N % ker_h != 0 || l2 % ker_w != 0 || l3 % ker_h != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	int i1, i2, i3;
	
	for(i3 = 0; i3 < M; i3 += l3)
		for(i2 = 0; i2 < N; i2 += l2)
			for(i1 = 0; i1 < K; i1 += l1)
				for(j = i2; j < i2+l2; j += ker_w)
					for(i = i3; i < i3+l3; i += ker_h)
						kernel(A, B, C, M, N, K, i, j, i1, i1+l1);
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

	matmul_v6(A,B,C_bis,M,N,K);
	
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

		
		
