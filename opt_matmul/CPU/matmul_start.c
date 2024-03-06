
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
	// Add your hown naive 3-loop implementation here
}


int main()
{
	size_t i;
	size_t M = 256, N = 256, K = 256;
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

	matmul_v1(A,B,C_bis,M,N,K);
	
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

		
		
