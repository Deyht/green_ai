
#include <stdlib.h>
#include <stdio.h>


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
	int i, j, k, dx, dy, dz;
	int v_size = 1024, rad = 3;
	float ***A, ***B;
	double acc;

	A = allocate_3D_array(v_size);
	B = allocate_3D_array(v_size);

	for(i = 0; i < v_size; i++)
		for(j = 0; j < v_size; j++)
			for(k = 0; k < v_size; k++)
				A[i][j][k] = i+j+k;

	// Add your hown naive 6-loop implementation here
	
	printf("%f\n", B[v_size/2][v_size/2][v_size/2]);

	exit(EXIT_SUCCESS);
}




