
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

float*** allocate_3D_array(size_t size)
{
	int i;
	float ***tab_3D;
	float **tab_2D;
	float *data;
	
	data = (float*) calloc(size*size*size, sizeof(float));
	tab_2D = (float**) malloc(size*size*sizeof(float*));
	tab_3D = (float***) malloc(size*sizeof(float**));
	
	for(i = 0; i < size*size; i++)
		tab_2D[i] = &(data[i*size]);

	for(i = 0; i < size; i++)
		tab_3D[i] = &(tab_2D[i*size]);
		
	return tab_3D;
}

int main()
{
	int ind, nb_threads, i, j, k, dx, dy, dz;
	int v_size = 1024, rad = 3;
	float ***A, ***B;
	double acc;

	A = allocate_3D_array(v_size);
	B = allocate_3D_array(v_size);

	for(i = 0; i < v_size; i++)
		for(j = 0; j < v_size; j++)
			for(k = 0; k < v_size; k++)
				A[i][j][k] = i+j+k;

	#pragma omp parallel shared(A, B, nb_threads) private(ind, j, k, dx, dy, dz, acc)
	{
		ind = omp_get_thread_num();
		if(ind == 0)
		{
			nb_threads = omp_get_num_threads();
			printf("There is currently %d threads running\n", nb_threads);
		}
		printf("Message from thread %d\n",ind);
				
		#pragma omp for schedule(dynamic, 1)
		for(i = rad; i < v_size-rad; i++)
			for(j = rad; j < v_size-rad; j++)
				for(k = rad; k < v_size-rad; k++)
				{
					acc = 0.0f;
					for(dx = i-rad; dx <= i+rad; dx++)
						for(dy = j-rad; dy <= j+rad; dy++)
							for(dz = k-rad; dz <= k+rad; dz++)
								acc += A[dx][dy][dz];
					B[i][j][k] = acc;
				}
	}
	
	printf("%f\n", B[v_size/2][v_size/2][v_size/2]);

	exit(EXIT_SUCCESS);
}
